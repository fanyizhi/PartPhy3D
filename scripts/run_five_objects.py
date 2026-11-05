"""默认五物体先导，支持先冻结队列或只测第一个对象。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile3d.experiments.five_object_pilot import main

if __name__ == "__main__":
    main()
