"""官方InstantMesh前端的只检查/显式执行入口。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile3d.research.frontend import main

if __name__ == "__main__":
    main()
