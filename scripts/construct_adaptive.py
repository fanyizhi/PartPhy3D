"""解析场上最终Mesh构造的稀疏/均匀对照。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile3d.research.adaptive_demo import main

if __name__ == "__main__":
    main()
