from pathlib import Path

from tactile3d.cli import make_demo_image
from tactile3d.config import load_config
from tactile3d.uncertainty.monte_carlo import run_inference


def test_demo_pipeline(tmp_path):
    image = tmp_path / "demo.png"
    make_demo_image(image)
    cfg = load_config(Path("configs/default.yaml"))
    cfg["num_samples"] = 2
    summary = run_inference(image, cfg, tmp_path / "out")
    assert summary["num_valid_samples"] == 2
    assert summary["object"]["volume_m3"]["mean"] > 0
    assert len(summary["parts"]) > 0
