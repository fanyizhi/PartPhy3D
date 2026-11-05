"""不使用体素化的Mesh构造状态学习诊断。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile3d.research.native_learning import main

if __name__ == "__main__":
    main()
