"""直接运行部件编辑；默认执行已知构造的CPU验证，无需模型或数据下载。"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    location = Path(__file__).resolve()
    project_root = next(
        (
            parent
            for parent in location.parents
            if (parent / "src/tactile3d/physics/edit_program.py").is_file()
        ),
        None,
    )
    if project_root is None:
        raise SystemExit("请把此脚本保存在项目根目录或项目scripts目录中")
    sys.path.insert(0, str(project_root / "src"))
    from tactile3d.experiments.edit_commands import main as run

    run()


if __name__ == "__main__":
    main()
