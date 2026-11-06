"""从公开ModelScope仓库下载固定文件版本，逐文件核验SHA-256。"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

CHUNK_BYTES = 16 * 1024 * 1024
WORKERS = 16
TIMEOUT_SECONDS = 90
RETRIES = 3
BLOCK_BYTES = 1024 * 1024
BASE_URL = "https://modelscope.cn"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            value.update(block)
    return value.hexdigest()


def read_metadata(url: str) -> dict[str, Any]:
    """元数据请求同样允许有限重试，避免已缓存权重因短时网络故障失效。"""
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as reply:
                return json.load(reply)
        except (OSError, json.JSONDecodeError) as error:
            if attempt == RETRIES - 1:
                raise RuntimeError("模型元数据请求失败，请保留缓存后重试") from error
            time.sleep(attempt + 1)
    raise RuntimeError("未取得模型元数据")


def clear_download_chunks(output: Path, name: str, size: int) -> None:
    """完整文件通过哈希后才清理本下载器创建的分块，控制磁盘峰值。"""
    parts = output / ".download_parts" / name.replace("/", "_")
    if not parts.is_dir():
        return
    for start in range(0, size, CHUNK_BYTES):
        for path in (parts / str(start), (parts / str(start)).with_suffix(".partial")):
            if path.is_file():
                path.unlink()
    if not any(parts.iterdir()):
        parts.rmdir()


def fetch(url: str, destination: Path, start: int, end: int, total: int) -> None:
    """校验Range响应，防止镜像忽略范围而把完整文件重复写入每一块。"""
    expected = end - start + 1
    if destination.is_file() and destination.stat().st_size == expected:
        return
    temporary = destination.with_suffix(".partial")
    for attempt in range(RETRIES):
        try:
            request_url = url
            if attempt and urllib.parse.urlparse(url).hostname == "modelscope.cn":
                separator = "&" if "?" in url else "?"
                request_url += (
                    f"{separator}download=true&range_start={start}"
                    f"&retry={attempt}&timestamp={time.time_ns()}"
                )
            request = urllib.request.Request(
                request_url,
                headers={"Range": f"bytes={start}-{end}", "Cache-Control": "no-cache"},
            )
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as reply:
                if total > CHUNK_BYTES:
                    required = f"bytes {start}-{end}/{total}"
                    if (
                        reply.status != 206
                        or reply.headers.get("Content-Range") != required
                    ):
                        raise ValueError("模型源没有返回约定的分块范围")
                received = 0
                with temporary.open("wb") as stream:
                    while block := reply.read(
                        min(BLOCK_BYTES, expected - received + 1)
                    ):
                        received += len(block)
                        if received > expected:
                            raise ValueError("下载内容超过约定长度")
                        stream.write(block)
                if received != expected:
                    raise ValueError("模型分块长度不完整")
            temporary.replace(destination)
            return
        except (OSError, ValueError) as error:
            if attempt == RETRIES - 1:
                raise RuntimeError(f"模型下载失败：{destination.name}") from error
            time.sleep(attempt + 1)


def download_file(model: str, metadata: dict[str, Any], output: Path) -> None:
    name = metadata["Path"]
    destination = (output / name).resolve()
    if not destination.is_relative_to(output.resolve()):
        raise ValueError("拒绝越出目标目录的模型路径")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = metadata["Sha256"]
    if destination.is_file():
        if digest(destination) != expected_hash:
            raise ValueError(f"已有文件哈希不符：{name}")
        clear_download_chunks(output, name, int(metadata["Size"]))
        print(f"VERIFIED {name}", flush=True)
        return
    size = int(metadata["Size"])
    if size <= 0 or len(expected_hash) != 64:
        raise ValueError("模型元数据缺少有效大小或SHA-256")
    url = metadata.get("DownloadURL") or (
        f"{BASE_URL}/models/{model}/resolve/{metadata['Revision']}/"
        f"{urllib.parse.quote(name)}"
    )
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError("公开权重或安装包下载必须使用HTTPS")
    parts = output / ".download_parts" / name.replace("/", "_")
    parts.mkdir(parents=True, exist_ok=True)
    tasks = [
        (start, min(size - 1, start + CHUNK_BYTES - 1))
        for start in range(0, size, CHUNK_BYTES)
    ]
    print(f"DOWNLOADING {name} {size} bytes {len(tasks)} chunks", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(fetch, url, parts / str(start), start, end, size): start
            for start, end in tasks
        }
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if count % 16 == 0 or count == len(tasks):
                print(f"PROGRESS {name} {count}/{len(tasks)}", flush=True)
    temporary = destination.with_suffix(destination.suffix + ".assembling")
    with temporary.open("wb") as stream:
        for start, _ in tasks:
            with (parts / str(start)).open("rb") as chunk:
                shutil.copyfileobj(chunk, stream, BLOCK_BYTES)
    if digest(temporary) != expected_hash:
        raise ValueError(f"模型文件完整性校验失败：{name}")
    temporary.replace(destination)
    clear_download_chunks(output, name, size)
    print(f"VERIFIED {name} sha256={expected_hash}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.model.split("/")) != 2 or ".." in args.model:
        parser.error("模型名称必须为organization/model")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "download_manifest.json"
    if metadata_path.is_file():
        payload = json.loads(metadata_path.read_text())
        if payload["model"] != args.model:
            raise ValueError("目标目录属于其他模型")
    else:
        url = (
            f"{BASE_URL}/api/v1/models/{args.model}/repo/files"
            "?Revision=master&Recursive=true"
        )
        metadata = read_metadata(url)
        if not metadata.get("Success"):
            raise RuntimeError("公开模型元数据请求失败")
        payload = {
            "model": args.model,
            "source": url,
            "files": metadata["Data"]["Files"],
        }
        metadata_path.write_text(json.dumps(payload, indent=2))
    verified = []
    for item in payload["files"]:
        if item["Type"] == "blob" and item["Path"] != ".gitattributes":
            download_file(args.model, item, args.output)
            verified.append(item["Path"])
    (args.output / "DOWNLOAD_COMPLETE.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "all_selected_runtime_files_sha256_verified": True,
                "verified_paths": verified,
                "selection": payload.get("selection", {"full_repository": True}),
                "omitted_metadata_paths": [".gitattributes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
