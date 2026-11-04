"""项目内质量对比与论文图入口，不执行任何训练。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from tactile3d.eval.benchmark import main as run

    run()


if __name__ == "__main__":
    main()
