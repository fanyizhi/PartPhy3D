"""固定构造的材料、示意色和物理状态同步编辑。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile3d.experiments.bound_material_demo import main

if __name__ == "__main__":
    main()
