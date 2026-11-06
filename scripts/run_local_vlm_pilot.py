"""本地权重的真实五物体图像推理入口。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile3d.experiments.local_vlm_pilot import main

if __name__ == "__main__":
    main()
