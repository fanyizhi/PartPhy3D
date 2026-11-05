from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tactile3d.utils import preflight


def test_failed_binary_import_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(name: str) -> None:
        raise OSError("binary extension unavailable")

    monkeypatch.setattr(preflight, "CORE_MODULES", {"test_binary": "test_binary"})
    monkeypatch.setattr(preflight.importlib, "import_module", fail_import)
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )
    result = preflight.check_installation(None)
    assert result["status"] == "failed"
    assert "binary extension unavailable" in result["errors"][0]
    assert result["gpu_execution_checked"] is False


def test_missing_lazy_training_api_causes_preflight_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight, "CORE_MODULES", {})
    monkeypatch.setattr(preflight, "TRAINING_MODULES", {"trl": "trl"})
    monkeypatch.setattr(preflight.sys, "platform", "win32")
    monkeypatch.setattr(
        preflight.importlib, "import_module", lambda name: SimpleNamespace()
    )
    monkeypatch.setattr(preflight, "version", lambda name: "fixture")
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )
    result = preflight.check_installation(None, training=True)
    assert result["status"] == "failed"
    assert "SFTConfig" in result["errors"][0]


def test_dataset_license_resources_ship_with_python_package(tmp_path: Path) -> None:
    from tactile3d.abo500.subset import write_attribution

    write_attribution(tmp_path)
    assert "CC BY 4.0" in (tmp_path / "ATTRIBUTION.md").read_text()
    assert "Creative Commons" in (tmp_path / "LICENSE-ABO-CC-BY-4.0.txt").read_text()
    assert (
        "Copyright (c) 2024 ajzhai"
        in (tmp_path / "LICENSE-NeRF2Physics-MIT.txt").read_text()
    )
