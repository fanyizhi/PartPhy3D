"""完整离线CPU验收，不访问服务器，不分配GPU。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile3d.research.suite import main

if __name__ == "__main__":
    main()
