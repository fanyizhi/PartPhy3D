"""Stage the released 30-view NeRF2Physics caches in the full benchmark directory."""

import argparse
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

from prepare_abo50_benchmark import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("remote")
    parser.add_argument("destination")
    parser.add_argument("--port", default="22")
    args = parser.parse_args()
    groups = read_json(args.dataset / "splits.json")
    ids = groups["val"] + groups["train"]
    ssh = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=20",
        "-p",
        args.port,
        args.remote,
    ]
    dest = shlex.quote(args.destination)
    subprocess.run(
        ssh + [f"mkdir -p {dest}/ready && cat > {dest}/cohort.json"],
        input=json.dumps({"object_ids": ids}).encode(),
        check=True,
    )
    # Send batches without keeping a second large archive on the workstation.
    for start in range(0, len(ids), 10):
        marker = f"{dest}/ready/{start:03d}"
        if subprocess.run(ssh + [f"test -f {marker}"]).returncode == 0:
            continue
        with subprocess.Popen(
            ssh + [f"cd {dest} && tar xzf - && touch ready/{start:03d}"],
            stdin=subprocess.PIPE,
        ) as remote:
            with tarfile.open(
                mode="w|gz", fileobj=remote.stdin, compresslevel=1
            ) as archive:
                for i in ids[start : start + 10]:
                    scene = args.dataset / "scenes" / i
                    files = [
                        scene / p
                        for p in (
                            "info.json",
                            "transforms.json",
                            "ns/point_cloud.ply",
                            "ns/dataparser_transforms.json",
                        )
                    ]
                    for sub in ("features", "images", "ns/renders/depth"):
                        files.extend(sorted((scene / sub).iterdir()))
                    for path in files:
                        archive.add(
                            path,
                            arcname=path.relative_to(args.dataset).as_posix(),
                            recursive=False,
                        )
            remote.stdin.close()
            if remote.wait():
                raise RuntimeError(f"Transfer stopped in batch {start}")
        print(f"Staged NeRF2Physics batch {start:03d}", flush=True)


if __name__ == "__main__":
    main()
