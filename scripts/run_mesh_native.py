"""Mesh原生物理积分、材料编辑和示意图入口。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile3d.experiments.mesh_native_demo import main

if __name__ == "__main__":
    main()
