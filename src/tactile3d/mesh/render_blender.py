from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from tactile3d.utils.io import ensure_dir


def render_mesh_views(
    mesh_path: str | Path,
    output_dir: str | Path,
    *,
    blender_executable: str = "blender",
    num_views: int = 12,
    resolution: int = 512,
    elevation_degrees: float = 20.0,
    overwrite: bool = False,
) -> Path:
    """Render a textured mesh through Blender's headless command line.

    Blender is kept outside the Python environment so the same entry point works
    on workstation and cluster installations. The bundled Blender script writes
    NeRF-style camera matrices consumed by the existing projection pipeline.
    """

    mesh_path = Path(mesh_path).expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh does not exist: {mesh_path}")
    if num_views < 1:
        raise ValueError("num_views must be positive")
    if resolution < 32:
        raise ValueError("resolution must be at least 32 pixels")

    output_dir = ensure_dir(Path(output_dir).expanduser().resolve())
    transforms = output_dir / "transforms.json"
    if transforms.is_file() and not overwrite:
        return transforms

    executable = shutil.which(blender_executable)
    if executable is None:
        candidate = Path(blender_executable).expanduser()
        if candidate.is_file():
            executable = str(candidate.resolve())
        else:
            raise FileNotFoundError(
                f"Blender executable not found: {blender_executable}. Install Blender or pass "
                "--blender /absolute/path/to/blender."
            )

    script_path = Path(__file__).with_name("blender_multiview.py")
    command = [
        executable,
        "--background",
        "--python",
        str(script_path),
        "--",
        "--mesh",
        str(mesh_path),
        "--output",
        str(output_dir),
        "--num-views",
        str(num_views),
        "--resolution",
        str(resolution),
        "--elevation-degrees",
        str(elevation_degrees),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not transforms.is_file():
        tail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(f"Blender rendering failed with code {completed.returncode}:\n{tail}")
    return transforms
