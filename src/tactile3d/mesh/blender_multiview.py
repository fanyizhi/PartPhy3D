"""Blender-side script. Run only through ``blender --background --python``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-views", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--elevation-degrees", type=float, default=20.0)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def _import_mesh(bpy, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    elif suffix == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=str(path))
        else:
            bpy.ops.import_mesh.ply(filepath=str(path))
    elif suffix == ".stl":
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=str(path))
        else:
            bpy.ops.import_mesh.stl(filepath=str(path))
    else:
        raise ValueError(f"Unsupported Blender mesh format: {suffix}")


def _look_at(camera, target, mathutils) -> None:
    direction = mathutils.Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    import bpy
    import mathutils

    args = _arguments()
    mesh_path = Path(args.mesh).resolve()
    output = Path(args.output).resolve()
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    _import_mesh(bpy, mesh_path)
    objects = [item for item in bpy.context.scene.objects if item.type == "MESH"]
    if not objects:
        raise RuntimeError(f"No mesh objects imported from {mesh_path}")

    corners = []
    for item in objects:
        corners.extend(item.matrix_world @ mathutils.Vector(corner) for corner in item.bound_box)
    low = mathutils.Vector(tuple(min(point[index] for point in corners) for index in range(3)))
    high = mathutils.Vector(tuple(max(point[index] for point in corners) for index in range(3)))
    center = (low + high) * 0.5
    # A 50 mm lens has a relatively narrow field of view. This distance keeps
    # the bounding-box diagonal inside a square frame at every azimuth.
    radius = max((high - low).length * 1.8, 1e-3)

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = 50.0
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    world = bpy.data.worlds.new("World")
    world.color = (0.8, 0.8, 0.8)
    bpy.context.scene.world = world
    light_data = bpy.data.lights.new("Key", type="AREA")
    light_data.energy = 1200.0
    light_data.shape = "DISK"
    light_data.size = radius
    light = bpy.data.objects.new("Key", light_data)
    bpy.context.collection.objects.link(light)
    light.location = center + mathutils.Vector((radius, -radius, radius))
    _look_at(light, center, mathutils)

    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    sensor_width = camera_data.sensor_width
    focal = args.resolution * camera_data.lens / sensor_width
    elevation = math.radians(args.elevation_degrees)
    frames = []
    for index in range(args.num_views):
        azimuth = 2.0 * math.pi * index / args.num_views
        horizontal = radius * math.cos(elevation)
        camera.location = center + mathutils.Vector(
            (horizontal * math.cos(azimuth), horizontal * math.sin(azimuth), radius * math.sin(elevation))
        )
        _look_at(camera, center, mathutils)
        relative = f"images/view_{index:03d}.png"
        scene.render.filepath = str(output / relative)
        bpy.ops.render.render(write_still=True)
        frames.append(
            {
                "file_path": relative,
                "transform_matrix": [list(row) for row in camera.matrix_world],
            }
        )

    payload = {
        "w": args.resolution,
        "h": args.resolution,
        "fl_x": focal,
        "fl_y": focal,
        "cx": args.resolution / 2.0,
        "cy": args.resolution / 2.0,
        "source_mesh": str(mesh_path),
        "frames": frames,
    }
    (output / "transforms.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
