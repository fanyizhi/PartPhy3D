"""构建源码ZIP、wheel、sdist及可选CPU结果包，不打包数据和密钥。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from build_release import build_release, digest


def package_project(root: Path, output: Path, results: Path | None) -> dict[str, str]:
    """先审计源码，再从审计副本构建安装包，避免递归打包运行目录。"""
    root, output = root.resolve(), output.resolve()
    if results is not None:
        results = results.resolve()
        if not results.is_relative_to(root / "outputs") or not results.is_dir():
            raise ValueError("结果目录必须位于本项目outputs内")
        suite = json.loads((results / "suite.json").read_text(encoding="utf-8"))
        if (
            suite.get("status") != "complete"
            or suite.get("scope") != "offline_CPU_diagnostics"
        ):
            raise ValueError("仅接受完整CPU验收结果，不自动打包任意运行日志")
    release = build_release(root, output)
    source = Path(release["source_directory"])
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(output / "dist"),
            str(source),
        ],
        check=True,
        shell=False,
    )
    if results is not None:
        with ZipFile(output / "PartPhy3D-CPU-results.zip", "x", ZIP_DEFLATED) as bundle:
            for path in sorted(results.rglob("*")):
                if path.is_file():
                    if path.is_symlink() or not path.resolve().is_relative_to(results):
                        raise ValueError("结果包不接受外部链接")
                    bundle.write(path, Path("results") / path.relative_to(results))
    artifacts = [path for path in output.iterdir() if path.is_file()]
    artifacts.extend((output / "dist").glob("*"))
    checksums = {
        path.relative_to(output).as_posix(): digest(path) for path in artifacts
    }
    (output / "SHA256SUMS.json").write_text(
        json.dumps(checksums, indent=2) + "\n", encoding="utf-8"
    )
    return checksums


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    try:
        checksums = package_project(
            Path(__file__).resolve().parents[1], args.output, args.results
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"打包失败：{error}\n")
    print(json.dumps(checksums, indent=2))


if __name__ == "__main__":
    main()
