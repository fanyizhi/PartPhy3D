"""离线CPU复现入口；从仓库运行，无需提前设置PYTHONPATH。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile3d.research.workflow import main

if __name__ == "__main__":
    main()
