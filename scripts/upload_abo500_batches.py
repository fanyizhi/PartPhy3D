"""Upload calibrated ABO inputs in sealed batches, reusing the official test run."""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.research.observed_cameras import calibrated_observed_cameras


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("remote")
    parser.add_argument("root")
    parser.add_argument("--port", default="22")
    args = parser.parse_args()
    splits = json.loads((args.dataset / "splits.json").read_text())
    ids = sum((splits[k] for k in ("train", "val", "test")), [])
    assert len(set(ids)) == len(ids) == 500
    ssh = ["ssh", "-o", "BatchMode=yes", "-p", args.port, args.remote]
    for split in ("val", "train"):
        for start in range(0, len(splits[split]), 25):
            scenes = splits[split][start : start + 25]
            dest = f"{args.root}/{split}_{start:03d}"
            if (
                subprocess.run(ssh + [f"test -f {dest}/observations/READY"]).returncode
                == 0
            ):
                continue
            subprocess.run(ssh + [f"mkdir -p {dest}/observations"], check=True)
            with tempfile.TemporaryFile() as stream:
                files, views = {}, {}
                with tarfile.open(fileobj=stream, mode="w") as archive:

                    def add(name, data):
                        info = tarfile.TarInfo(name)
                        info.size = len(data)
                        archive.addfile(info, io.BytesIO(data))

                    for scene in scenes:
                        cam = f"scenes/{scene}/transforms.json"
                        camera = json.loads((args.dataset / cam).read_bytes())
                        views[scene] = calibrated_observed_cameras(camera)[0].tolist()
                        names = [cam]
                        for i in sorted(set(views[scene]) | {0, 8, 20}):
                            frame = f"images/{scene}_{i:02d}.png"
                            assert camera["frames"][i]["file_path"] == frame
                            names.append(f"scenes/{scene}/{frame}")
                        for name in names:
                            data = (args.dataset / name).read_bytes()
                            files[name] = hashlib.sha256(data).hexdigest()
                            add(name, data)
                    add(
                        "cohort.json",
                        json.dumps(
                            dict(
                                object_ids=scenes,
                                official_split=split,
                                reconstruction_views=views,
                                mass_views=[0, 8, 20],
                                files_sha256=files,
                            )
                        ).encode(),
                    )
                stream.seek(0)
                subprocess.run(
                    ssh + [f"cd {dest}/observations && tar xf - && touch READY"],
                    stdin=stream,
                    check=True,
                )
            print(dest, len(scenes), flush=True)


if __name__ == "__main__":
    main()
