"""Freeze a deterministic 50-object ABO-500 benchmark and package original views."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(chunk)
        return checksum.hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def prepare(root: Path, published: Path, output: Path) -> None:
    split_path = root / "splits.json"
    official = read_json(split_path)["test"]
    protocol = read_json(published)
    if (
        len(official) != 100
        or len(set(official)) != 100
        or set(official) != set(protocol["object_ids"])
    ):
        raise ValueError("Local and published test sets must agree on 100 distinct IDs")
    development = set(protocol["previously_exposed_development_ids"])
    development_products = {scene.split("_")[0] for scene in development}
    eligible = [
        scene for scene in official if scene.split("_")[0] not in development_products
    ]
    salt = "partphy3d-research1-abo50-v1:"
    ranked = sorted(
        eligible, key=lambda scene: hashlib.sha256((salt + scene).encode()).hexdigest()
    )
    selected = ranked[:50]
    if len(selected) != 50:
        raise ValueError("Fewer than 50 eligible test scenes")

    files = []
    scenes = []
    for scene in selected:
        directory = root / "scenes" / scene
        camera_path = directory / "transforms.json"
        frames = read_json(camera_path)["frames"]
        expected = [f"images/{scene}_{index:02d}.png" for index in range(30)]
        if [frame["file_path"] for frame in frames] != expected:
            raise ValueError(f"Unexpected calibrated frame order in {scene}")
        image_paths = [directory / PurePosixPath(path) for path in expected]
        files.extend([camera_path, *image_paths])
        scenes.append(
            {
                "scene_id": scene,
                "single_view": image_paths[8].relative_to(root).as_posix(),
                "multiview_count": len(image_paths),
            }
        )
    hashes = {path.relative_to(root).as_posix(): digest(path) for path in files}

    output.mkdir(parents=True, exist_ok=False)
    cohort = {
        "cohort_name": "partphy3d_abo50_test_v1",
        "object_ids": selected,
        "required_original_split": "test",
        "source_split_sha256": digest(split_path),
        "selection": "First 50 by SHA-256 of salt + scene ID; no image or label filtering",
        "selection_salt": salt,
        "eligible_count": len(eligible),
        "excluded_development_products": sorted(development_products),
        "single_view_zero_based_index": 8,
        "single_view_reason": "Predeclared index used in the existing PUGS-Qwen adapter",
        "multiview_zero_based_indices": list(range(30)),
        "scenes": scenes,
        "files_sha256": hashes,
    }
    write_json(output / "cohort.json", cohort)
    with tarfile.open(output / "observations.tar", "w") as archive:
        archive.add(output / "cohort.json", arcname="cohort.json")
        for path in files:
            archive.add(path, arcname=path.relative_to(root).as_posix())
    print(f"Frozen {len(selected)} objects and {len(selected) * 30} original images")
    print(
        f"Archive size: {(output / 'observations.tar').stat().st_size / 2**20:.1f} MiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--published", type=Path, default=Path("configs/abo500_mass_test.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.published, args.output)


if __name__ == "__main__":
    main()
