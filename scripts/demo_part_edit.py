"""在项目根目录或任意目录启动本地部件编辑验证。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from tactile3d.experiments.part_edit_demo import main as run

    run()


if __name__ == "__main__":
    main()
