"""安装后只读检查依赖导入及内置数据，不加载模型或分配GPU张量。"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

CORE_MODULES = {
    "tactile3d": "tactile3d",
    "numpy": "numpy",
    "Pillow": "PIL",
    "PyYAML": "yaml",
    "trimesh": "trimesh",
    "manifold3d": "manifold3d",
    "fast-simplification": "fast_simplification",
    "shapely": "shapely",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
}
TRAINING_MODULES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "open_clip_torch": "open_clip",
    "transformers": "transformers",
    "datasets": "datasets",
    "accelerate": "accelerate",
    "peft": "peft",
    "trl": "trl",
}


def check_installation(
    data_root: Path | None, training: bool = False
) -> dict[str, Any]:
    """检查实际可导入的模块、训练API及包依赖一致性。

    Args:
        data_root: 五物体目录；None只检查软件。
        training: 是否同时导入训练库；不执行模型推理或GPU测试。

    Returns:
        软件、数据检查及错误清单。GPU训练能力需在服务器另行验收。
    """
    modules = {**CORE_MODULES, **(TRAINING_MODULES if training else {})}
    if training and sys.platform.startswith("linux"):
        modules["bitsandbytes"] = "bitsandbytes"
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "gpu_execution_checked": False,
        "modules": {},
        "errors": [],
    }
    for package, module_name in modules.items():
        try:
            module = importlib.import_module(module_name)
            report["modules"][package] = version(package)
            if package == "transformers":
                for name in ("AutoModelForImageTextToText", "AutoProcessor"):
                    getattr(module, name)
            if package == "trl":
                for name in ("SFTConfig", "SFTTrainer"):
                    getattr(module, name)
        except (
            ImportError,
            OSError,
            RuntimeError,
            AttributeError,
            ValueError,
        ) as error:
            # 二进制扩展可能抛OSError，延迟导入也可能抛RuntimeError，需统一报告。
            report["errors"].append(f"{package}: {type(error).__name__}: {error}")
    dependency_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    if dependency_check.returncode:
        report["errors"].append(dependency_check.stdout + dependency_check.stderr)
    if data_root is not None:
        try:
            from tactile3d.abo500.subset import verify_subset

            subset = verify_subset(data_root)
            report["data"] = {
                "root": str(data_root.resolve()),
                "objects": subset["num_objects"],
                "views": subset["num_views"],
                "sha256": "passed",
            }
        except (OSError, ValueError, TypeError, KeyError) as error:
            report["errors"].append(f"dataset: {error}")
    report["status"] = "passed" if not report["errors"] else "failed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data/abo500_five"))
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args()
    report = check_installation(
        None if args.skip_data else args.data_root, args.training
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
