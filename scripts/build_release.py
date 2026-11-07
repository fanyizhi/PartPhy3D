"""构建源码与固定重放资产包，不含完整数据、模型权重或本机SSH设置。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

SOURCE_DIRECTORIES = (
    "src",
    "scripts",
    "configs",
    "examples",
    "tests",
    "docs",
    ".github",
)
ROOT_FILES = (
    "README.md",
    "README_zh.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "THIRD_PARTY.md",
    "pyproject.toml",
    "requirements.txt",
    "environment.yml",
    ".gitignore",
    ".gitattributes",
    "MANIFEST.in",
    "vendor/README.md",
    "vendor/SOURCES.json",
    "vendor/LICENSE.txt",
    "vendor/LICENSE_cctorch.txt",
    "vendor/InstantMesh_LICENSE.txt",
)
BINARY_FILES = {
    "vendor/wheels/sam_2-1.0-py3-none-any.whl": (
        "7d88f9df712bf39fce96af4dfd6cf238f812b5c992affffba7296a2671cccaee"
    ),
    "examples/observed_surface_demo/generated_camera_space.ply": (
        "96e1895cdd28b4e84e8210c501572ddea934f3efc2dc564d2e52bb845b16fe0c"
    ),
    "examples/observed_surface_demo/B013TGFGWS_A1F83G8C2ARO7P/"
    "surface_fields.npz": (
        "ca2086a8452e8e8b8f752fe3b0662c8e4b1b776efc0af0429f4c5fdce0c12d9c"
    ),
    "examples/observed_surface_demo/B013TGFGWS_A1F83G8C2ARO7P/"
    "view_00.npz": ("5d7d326eb49dd890e32f59e0450529aabb6429c2ec59640d21a364539e523ed4"),
}
ALLOWED_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".sh",
    ".png",
    ".svg",
    ".txt",
    ".csv",
    ".html",
}
EXCLUDED_FILES = {
    "scripts/verify_server_preflight.py",
    "scripts/build_visio_figures.py",
}
MAX_FILE_BYTES = 5 * 1024 * 1024
SECRET_PATTERN = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}"
    r"|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{25,}"
    r"|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----)"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_files(root: Path) -> list[Path]:
    """先校验完整清单，再允许写入新包，避免误将运行资源纳入发布。

    Args:
        root: 明确的项目根目录。

    Returns:
        按相对路径排序、无符号链接的源码与文档文件。
    """
    files = [root / name for name in (*ROOT_FILES, *BINARY_FILES)]
    for directory in SOURCE_DIRECTORIES:
        files.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and path.suffix in ALLOWED_SUFFIXES
            and "__pycache__" not in path.parts
            and not any(
                part.endswith((".egg-info", ".dist-info")) for part in path.parts
            )
            and path.relative_to(root).as_posix() not in EXCLUDED_FILES
        )
    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"发布不接受外部路径或符号链接：{relative}")
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"发布文件缺失或超过大小预算：{relative}")
        if relative in BINARY_FILES:
            if digest(path) != BINARY_FILES[relative]:
                raise ValueError(f"固定二进制资产校验失败：{relative}")
        elif path.suffix != ".png":
            content = path.read_text(encoding="utf-8-sig")
            if SECRET_PATTERN.search(content):
                raise ValueError(f"检测到疑似凭据，停止发布：{relative}")
            if path.suffix == ".py":
                ast.parse(content, filename=relative, feature_version=(3, 10))
            if path.suffix == ".json":
                json.loads(content)
    return sorted(set(files))


def build_release(root: Path, output: Path) -> dict[str, Any]:
    """导出新目录、SHA256清单和ZIP；不访问GitHub或执行训练。

    Args:
        root: 项目源码目录。
        output: 项目outputs目录内尚不存在的目标目录。

    Returns:
        包位置、文件数量、大小与校验值。
    """
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root / "outputs"):
        raise ValueError("发布输出必须位于项目outputs目录内")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有发布：{output}")
    files = release_files(root)
    destination = output / "PartPhy3D"
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "project": "PartPhy3D",
        "research_status": "development_prototype",
        "full_scientific_pipeline_complete": False,
        "excluded": [
            "data",
            "outputs",
            "checkpoints",
            "caches",
            *sorted(EXCLUDED_FILES),
        ],
        "files": {},
    }
    for source in files:
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".png" or relative.as_posix() in BINARY_FILES:
            shutil.copy2(source, target)
        else:
            # 统一导出换行，避免Git规范化后与发布哈希不一致。
            target.write_text(
                source.read_text(encoding="utf-8-sig"),
                encoding="utf-8",
                newline="\n",
            )
        manifest["files"][relative.as_posix()] = digest(target)
    manifest_file = destination / "RELEASE_MANIFEST.json"
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    archive = output / "PartPhy3D-source.zip"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as bundle:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output).as_posix())
    with ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("ZIP完整性校验失败")
    return {
        "source_directory": str(destination),
        "archive": str(archive),
        "file_count": len(files) + 1,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": digest(archive),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or root / "outputs/releases" / f"run_{time.time_ns()}"
    try:
        summary = build_release(root, output)
    except (OSError, ValueError, SyntaxError) as error:
        parser.exit(1, f"发布准备失败：{error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
