"""打包项目代码和固定五物体，供Linux服务器一次解压后安装。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from build_release import build_release, digest

from tactile3d.abo500.subset import MANIFEST_NAME, verify_subset
from tactile3d.experiments.five_object_pilot import read_cohort
from tactile3d.physics.part_io import write_json

DATA_PATH = Path("data/abo500_five")
BUNDLE_NAME = "PartPhy3D-with-data.zip"


def package_server(root: Path, output: Path) -> dict[str, Any]:
    """将审核后的源码与校验后的五物体合并，不引入其他data或运行目录。

    Args:
        root: 项目根目录，包含固定队列和已准备好的子集。
        output: 项目outputs内部不存在的目录。

    Returns:
        完整包位置、大小、SHA-256及数据数量；第三方库由requirements安装。
    """
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root / "outputs"):
        raise ValueError("迁移包必须输出到项目outputs目录")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有迁移包：{output}")
    dataset = root / DATA_PATH
    manifest = verify_subset(dataset)
    selection = read_cohort(root / "configs/five_object_pilot.json")
    if manifest["object_ids"] != selection["object_ids"]:
        raise ValueError("数据子集与固定队列不一致")
    release = build_release(root, output / "audited_source")
    destination = Path(release["source_directory"])
    source_manifest = json.loads(
        (destination / "RELEASE_MANIFEST.json").read_text(encoding="utf-8")
    )
    hashes = dict(source_manifest["files"])
    hashes["RELEASE_MANIFEST.json"] = digest(destination / "RELEASE_MANIFEST.json")
    # 用已核验清单限定复制集合，避免把同级数据、缓存或本机私有文件带走。
    for relative in [*manifest["files"], MANIFEST_NAME]:
        source = dataset / relative
        target = destination / DATA_PATH / relative
        if source.is_symlink() or not source.resolve().is_relative_to(dataset):
            raise ValueError("数据迁移不接受目录外链接")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        expected = digest(source)
        if digest(target) != expected:
            raise ValueError(f"数据复制核对失败：{relative}")
        hashes[(DATA_PATH / relative).as_posix()] = expected
    write_json(
        destination / "SERVER_MANIFEST.json",
        {
            "schema": "partphy3d_server_bundle_v1",
            "files": hashes,
            "data_root": DATA_PATH.as_posix(),
            "objects": manifest["num_objects"],
            "views": manifest["num_views"],
            "python_dependencies_bundled": False,
            "install_command": "python -m pip install -r requirements.txt",
            "training_started": False,
        },
    )
    archive = output / BUNDLE_NAME
    with ZipFile(archive, "x", ZIP_DEFLATED) as bundle:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                bundle.write(path, Path("PartPhy3D") / path.relative_to(destination))
    with ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("迁移包CRC核对失败")
    checksum = digest(archive)
    (output / f"{BUNDLE_NAME}.sha256").write_text(
        f"{checksum}  {BUNDLE_NAME}\n", encoding="ascii"
    )
    return {
        "archive": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": checksum,
        "objects": manifest["num_objects"],
        "views": manifest["num_views"],
        "status": "complete",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "outputs/server_bundle")
    args = parser.parse_args()
    try:
        report = package_server(root, args.output)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"服务器迁移包构建失败：{error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
