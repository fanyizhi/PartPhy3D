"""Source releases must not collect untracked local experiment leftovers."""

import importlib.util
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def release_root(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/build_release.py"
    spec = importlib.util.spec_from_file_location("source_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT_FILES", ("README.md",))
    monkeypatch.setattr(module, "BINARY_FILES", {})
    monkeypatch.setattr(module, "SOURCE_DIRECTORIES", ("src",))
    (tmp_path / "README.md").write_text("Source release")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.py").write_text("VALUE = 1\n")
    return tmp_path, module


def test_git_release_excludes_untracked_experiment_files(release_root):
    root, module = release_root
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "add", "README.md", "src/main.py"], check=True
    )
    (root / "src/scratch.py").write_text("EXPERIMENT = True\n")
    assert {p.relative_to(root).as_posix() for p in module.release_files(root)} == {
        "README.md",
        "src/main.py",
    }


def test_exported_source_tree_can_package_without_git(release_root):
    root, module = release_root
    assert len(module.release_files(root)) == 2


def test_repository_does_not_track_raw_data_or_run_outputs():
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("Repository tracking check requires a Git checkout")
    tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"])
    names = [name for name in tracked.decode().split("\0") if name]
    forbidden = [
        name
        for name in names
        if (
            name.startswith(("outputs/", "checkpoints/", "wandb/"))
            or (
                name.startswith("data/")
                and Path(name).suffix.lower()
                in {".png", ".jpg", ".jpeg", ".glb", ".obj", ".ply"}
            )
            or Path(name).suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}
        )
    ]
    assert not forbidden, f"Move generated resources outside Git: {forbidden}"
    figure_sets = {
        Path(name).parent.as_posix()
        for name in names
        if name.startswith("paper_assets/") and Path(name).suffix in {".png", ".svg"}
    }
    assert len(figure_sets) <= 1, "Keep only the current figure set in Git"
