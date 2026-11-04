"""无需数据或GPU的编辑观测信息解析审计。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from tactile3d.experiments.observability import main as run

    run()


if __name__ == "__main__":
    main()
