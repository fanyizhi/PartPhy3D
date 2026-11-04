from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from tactile3d.research.frontend import execute_instantmesh, instantmesh_plan


def fixture_plan(tmp_path: Path) -> dict[str, Any]:
    repository = tmp_path / "model repo"
    (repository / "configs").mkdir(parents=True)
    (repository / "run.py").write_text("print('protocol_fixture')\n", encoding="utf-8")
    (repository / "configs/instant-mesh-large.yaml").write_text(
        "seed: 17\n", encoding="utf-8"
    )
    image = tmp_path / "source.with.dots.png"
    Image.new("RGB", (16, 16), (120, 80, 40)).save(image)
    return instantmesh_plan(
        repository,
        Path(sys.executable),
        image,
        tmp_path / "output",
        0.125,
        "independently measured calibration target",
    )


def test_frontend_check_is_read_only_and_keeps_arguments_separate(
    tmp_path: Path,
) -> None:
    plan = fixture_plan(tmp_path)
    assert not Path(plan["output"]).exists()
    assert plan["command"][0] == str(Path(sys.executable).resolve())
    assert plan["command"][1].endswith("run.py")
    assert plan["expected_mesh"].endswith("observed.obj")
    assert plan["generated_views_are_observations"] is False


def test_missing_model_and_invalid_scale_do_not_fallback(tmp_path: Path) -> None:
    plan = fixture_plan(tmp_path)
    with pytest.raises(ValueError, match="尺度"):
        instantmesh_plan(
            Path(plan["repository"]),
            Path(sys.executable),
            Path(plan["image"]),
            tmp_path / "other",
            0,
            "",
        )
    Path(plan["repository"], "run.py").unlink()
    with pytest.raises(FileNotFoundError):
        instantmesh_plan(
            Path(plan["repository"]),
            Path(sys.executable),
            Path(plan["image"]),
            tmp_path / "other",
            1,
            "reference",
        )


def test_failed_external_process_is_retained_and_not_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from subprocess import CompletedProcess

    plan = fixture_plan(tmp_path)

    def failed_process(command: list[str], **kwargs: Any) -> CompletedProcess[str]:
        assert kwargs["shell"] is False
        return CompletedProcess(command, 7)

    monkeypatch.setattr("tactile3d.research.frontend.subprocess.run", failed_process)
    with pytest.raises(RuntimeError, match="退出码7"):
        execute_instantmesh(plan)
    status = json.loads(
        Path(plan["output"], "frontend.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"
    assert not Path(plan["output"], "metric_mesh.glb").exists()


def test_changed_observation_is_rejected_before_execution(tmp_path: Path) -> None:
    plan = fixture_plan(tmp_path)
    Path(plan["image"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="图像"):
        execute_instantmesh(plan)
    assert not Path(plan["output"]).exists()
