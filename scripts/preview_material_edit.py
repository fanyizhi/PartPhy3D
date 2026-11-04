"""运行ABO脚凳的材料颜色替换预览。"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from tactile3d.experiments.material_preview import main as run

    run()


if __name__ == "__main__":
    main()
