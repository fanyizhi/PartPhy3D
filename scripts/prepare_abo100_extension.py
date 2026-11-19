"""Freeze the official ABO test set and package only the new cases' required views."""

import argparse
import sys
import tarfile
from pathlib import Path

from prepare_abo50_benchmark import digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile3d.research.observed_cameras import calibrated_observed_cameras


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "previous", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    previous = read_json(args.previous)["object_ids"]
    split = args.source / "splits.json"
    official = read_json(split)["test"]
    declared = read_json(ROOT / "configs/abo500_mass_test.json")
    if (
        len(official) != 100
        or len(set(official)) != 100
        or set(official) != set(declared["object_ids"])
    ):
        raise ValueError("Expected the official 100 distinct test objects")
    if (
        len(previous) != 50
        or len(set(previous)) != 50
        or not set(previous) <= set(official)
    ):
        raise ValueError(
            "Previous cases must be the 50-object subset of the official test set"
        )
    added = [s for s in official if s not in previous]
    mass_views = (0, 8, 20)
    files = {}
    mesh_views = {}
    for scene in added:
        camera = f"scenes/{scene}/transforms.json"
        payload = read_json(args.source / camera)
        indices, _, _ = calibrated_observed_cameras(payload)
        mesh_views[scene] = indices.tolist()
        files[camera] = digest(args.source / camera)
        for i in sorted(set(mesh_views[scene]) | set(mass_views)):
            frame = f"images/{scene}_{i:02d}.png"
            if payload["frames"][i]["file_path"] != frame:
                raise ValueError(f"Frame filename mismatch: {scene}, {i}")
            relative = f"scenes/{scene}/{frame}"
            files[relative] = digest(args.source / relative)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "cohort100.json",
        dict(
            object_ids=official,
            prior_object_ids=previous,
            additional_object_ids=added,
            split_sha256=digest(split),
            prior_cohort_sha256=digest(args.previous),
            historical_development_ids=declared["previously_exposed_development_ids"],
        ),
    )
    manifest = args.output / "extension_cohort.json"
    write_json(
        manifest,
        dict(
            object_ids=added,
            files_sha256=files,
            reconstruction_views=mesh_views,
            mass_views=mass_views,
        ),
    )
    with tarfile.open(args.output / "observations.tar", "w") as archive:
        archive.add(manifest, arcname="cohort.json")
        for relative in files:
            archive.add(args.source / relative, arcname=relative)
    print(
        f"Packaged {len(added)} objects, {len(files) - len(added)} raw images",
        flush=True,
    )


if __name__ == "__main__":
    main()
