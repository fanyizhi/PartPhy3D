"""Multi-round VLM construction hypotheses integrated over fixed metric cells."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tactile3d.eval.benchmark import verify_prediction_seal
from tactile3d.experiments.ten_objects import depth_within_region, load_mesh
from tactile3d.figures.mesh_fields import render_mesh
from tactile3d.mesh.assembly import texture_face_rgb
from tactile3d.models.calibration import mass_metrics
from tactile3d.physics.fields import integrate_density_cells
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.vision.local_vlm import LocalVLM
from tactile3d.vision.view_selection import select_azimuth_views
from tactile3d.vision.vlm_interior import _parse_response_content


def integrate_construction(
    points, owners, pitch, rows, densities, thicknesses=None, thickness_bounds=None
):
    """Allocate each disjoint cell once; material fractions refer to core volume."""
    points, owners = np.asarray(points), np.asarray(owners)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(points)
        or not np.isfinite(points).all()
        or owners.shape != (len(points),)
        or owners.dtype.kind not in "iu"
        or np.any(owners < 0)
        or not np.isfinite(pitch)
        or pitch <= 0
        or not densities
        or any(not np.isfinite(rho) or rho <= 0 for rho in densities.values())
    ):
        raise ValueError("Invalid metric cells or density library")
    requested = set(np.unique(owners).tolist())
    if len(rows) != len(requested) or {r["region_id"] for r in rows} != requested:
        raise ValueError("VLM must cover every occupied region exactly once")
    depth = depth_within_region(points, pitch)
    density = np.zeros(len(points))
    occupied = np.zeros(len(points))
    material_volumes = dict.fromkeys(densities, 0.0)
    for row in rows:
        surface = row["surface_material"]
        thickness = float(row["wall_thickness_m"])
        if surface not in densities or not np.isfinite(thickness) or thickness < 0:
            raise ValueError("Invalid shell material or thickness")
        if thickness_bounds is not None and thickness > 0:
            low, high = thickness_bounds[surface]
            if not low <= thickness <= high:
                raise ValueError(
                    f"{surface} shell thickness {thickness} m outside "
                    f"material bounds [{low}, {high}] m"
                )
        if thicknesses is not None and thickness > 0:
            thickness = float(thicknesses[surface])
            if not np.isfinite(thickness) or thickness < 0:
                raise ValueError("Invalid library thickness")
        phases = row["core"]
        fractions = np.asarray([p["fraction"] for p in phases], dtype=float)
        if (
            not np.isfinite(fractions).all()
            or np.any(fractions < 0)
            or fractions.sum() > 1 + 1e-8
            or any(p["material"] not in densities for p in phases)
        ):
            raise ValueError(
                "Core fractions must be finite nonnegative volume fractions summing <=1"
            )
        selected = owners == row["region_id"]
        shell = np.clip(
            (thickness - np.maximum(depth[selected] - pitch, 0)) / pitch, 0, 1
        )
        core_density = sum(p["fraction"] * densities[p["material"]] for p in phases)
        density[selected] = shell * densities[surface] + (1 - shell) * core_density
        occupied[selected] = shell + (1 - shell) * fractions.sum()
        material_volumes[surface] += float(shell.sum() * pitch**3)
        core_volume = float((1 - shell).sum() * pitch**3)
        for phase in phases:
            material_volumes[phase["material"]] += core_volume * phase["fraction"]
    moments = integrate_density_cells(points, density, pitch, owners)
    material_mass = sum(densities[k] * v for k, v in material_volumes.items())
    if not np.isclose(material_mass, moments["mass_kg"], rtol=1e-10):
        raise ValueError("Material allocation and cell mass disagree")
    if sum(material_volumes.values()) > len(points) * pitch**3 * (1 + 1e-8):
        raise ValueError("Material allocations exceed the envelope")
    return {
        "moments": moments,
        "material_volumes_m3": material_volumes,
        "occupied_volume_m3": float(occupied.sum() * pitch**3),
    }


def prepare_evidence(root, assets, geometry, scene, output):
    source = root / "scenes" / scene
    mesh = load_mesh(assets / scene / "model.glb")
    mesh.vertices = mesh.vertices[:, [0, 2, 1]] * [1, -1, 1]
    labels = np.load(geometry / "objects" / scene / "face_labels.npy")
    rgb = texture_face_rgb(mesh)
    count = int(labels.max()) + 1
    palette = np.asarray(
        [
            [0.15, 0.45, 0.7],
            [0.9, 0.4, 0.1],
            [0.2, 0.65, 0.4],
            [0.7, 0.3, 0.6],
            [0.6, 0.55, 0.1],
            [0.25, 0.7, 0.75],
            [0.55, 0.4, 0.25],
            [0.4, 0.4, 0.6],
        ]
    )
    overview = render_mesh(
        mesh.vertices, mesh.faces, palette[labels % len(palette)], resolution=600
    )
    Image.fromarray(np.asarray(overview, dtype=np.uint8)).save(output / "regions.png")
    sheet = Image.new("RGB", (1000, 500), "white")
    draw = ImageDraw.Draw(sheet)
    for region in range(count):
        part = mesh.submesh(
            [np.flatnonzero(labels == region)], append=True, repair=False
        )
        panel = render_mesh(
            part.vertices, part.faces, rgb[labels == region], resolution=220
        )
        x, y = region % 4 * 250, region // 4 * 250
        sheet.paste(Image.fromarray(np.asarray(panel, dtype=np.uint8)), (x, y + 24))
        draw.text((x + 8, y + 5), f"REGION {region}", fill="black")
    sheet.save(output / "numbered_regions.png")
    transforms = json.loads((source / "transforms.json").read_text())
    selected = select_azimuth_views(
        np.asarray([f["transform_matrix"] for f in transforms["frames"]]), 4
    )
    paths = [
        source / transforms["frames"][v["frame_index"]]["file_path"] for v in selected
    ]
    paths += [output / "regions.png", output / "numbered_regions.png"]
    write_json(
        output / "evidence.json",
        {
            "views": selected,
            "files": {str(p): file_sha256(p) for p in paths},
            "mesh_sha256": file_sha256(assets / scene / "model.glb"),
        },
    )
    return [Image.open(p).copy() for p in paths]


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    verify_prediction_seal(args.geometry)
    ids = json.loads(args.cohort.read_text(encoding="utf-8"))["object_ids"]
    spec = {
        "cohort": ids,
        "rounds": 3,
        "surface_hints": args.surface_hints,
        "thickness_policy": (
            "unconstrained_ablation"
            if args.allow_unconstrained_thickness
            else "material_bounds"
        ),
        "seed": 1200,
        "geometry_source": str(args.geometry),
        "model": str(args.model),
        "code_sha256": file_sha256(Path(__file__)),
        "mass_labels_used_for_inference": False,
        "aggregation": "equal mean of three whole-object construction integrals",
        "uncertainty": "correlated self-check rounds, not calibrated posterior samples",
    }
    protocol = args.output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != spec:
        raise ValueError("Saved run uses a different frozen protocol")
    write_json(protocol, spec)
    model = LocalVLM(args.model, max_tokens=2600, max_pixels=384**2, quantized=True)
    write_json(args.output / "model.json", model.provenance)
    predictions = []
    for scene in ids:
        directory = args.output / scene
        directory.mkdir(exist_ok=True)
        images = prepare_evidence(
            args.root, args.assets, args.geometry, scene, directory
        )
        original = json.loads(
            (args.geometry / "objects" / scene / "legacy_prediction.json").read_text()
        )
        densities = {
            m["name"]: float(np.mean(m["density"])) for m in original["materials"]
        }
        bounds = {m["name"]: m["thickness_m"] for m in original["materials"]}
        checked_bounds = None if args.allow_unconstrained_thickness else bounds
        with np.load(
            args.geometry / "objects" / scene / "nearest_surface_field.npz"
        ) as field:
            points, owners, pitch = (
                field["centers_m"],
                field["part_ids"],
                float(field["pitch_m"]),
            )
        volumes = {
            int(i): float((owners == i).sum() * pitch**3) for i in np.unique(owners)
        }
        prompt = (
            "First four images are actual views of ONE object. Image 5 colors geometric regions; "
            "image 6 shows isolated textured regions with their authoritative REGION IDs. "
            "Regions may cross functional parts. Infer construction of each occupied region. "
            "Do not estimate mass. Use ONLY these material names/densities (kg/m^3): "
            f"{json.dumps(densities)}. Envelope volumes in m^3: {json.dumps(volumes)}. "
            "Allocate volume, not material-name confidence. A metal screw candidate is NOT an "
            "entire metal region. Cloth covers are thin; foam and hidden frame occupy the core. "
            "Containers and bags can contain empty space. Distinguish coated dense cores from "
            "hollow tubes. A solid uses thickness 0 and core material fraction 1. A hollow shell "
            "uses its physical wall thickness and empty core []. Layered structures use "
            "surface material/thickness and core phases with volumetric fractions summing <=1; "
            "remaining fraction is void. Do not fill a geometric wood-colored region entirely "
            "with wood unless construction supports it. Do not copy example values. "
            "Return JSON {regions:[{region_id:integer,construction:string,surface_material:string,"
            "wall_thickness_m:number,core:[{material:string,fraction:number}],reason:string}]}. "
            "Cover every listed region exactly once. Give concise reasons. Hidden structure is inferred."
        )
        results = []
        if checked_bounds is not None:
            prompt += (
                "\nMANDATORY material-specific shell thickness bounds in METERS: "
                + json.dumps(bounds)
                + ". Every positive wall_thickness_m must lie in "
                "its surface material's bounds. Cloth covering, foam and internal frame "
                "are separate layers. Do not use total cushion thickness for cloth. "
                "Thickness 0 denotes no distinct shell, with construction encoded in core."
            )
        if args.surface_hints:
            hints = {
                r["part_id"]: r["material_hypothesis_probabilities_uncalibrated"]
                for r in original["parts"]
            }
            prompt += (
                "\nIndependent weak surface evidence from texture/color for each numbered region: "
                + json.dumps(hints)
                + ". These are uncalibrated surface-material scores, "
                "NOT core volume fractions. Use them to resolve region identity/material binding "
                "with the images; explicitly justify contradictions. A geometric region can "
                "include several physical pieces. Do not relabel all regions as the same frame "
                "or leg without matching the numbered textured patches."
            )
        previous = None
        for trial in range(3):
            path = directory / f"round_{trial}.json"
            question = prompt
            if trial:
                question += (
                    "\nReview this previous construction hypothesis against the images and region "
                    "sizes. Check especially thin covers, hollow spaces, frame location and "
                    "volume fractions. Revise unsupported assignments; do not infer a target mass. "
                    + json.dumps(previous)
                )
            if path.exists():
                record = json.loads(path.read_text())
                if record["prompt"] != question:
                    raise ValueError("Cached prompt differs from frozen evidence")
                answer = record["answer"]
            else:
                error = None
                for attempt in range(2):
                    query = (
                        question
                        if error is None
                        else (
                            question
                            + "\nYour previous JSON failed validation: "
                            + str(error)
                            + ". Return a complete corrected JSON with valid volume fractions."
                        )
                    )
                    response = model.generate(
                        query,
                        images,
                        "Return physically consistent construction JSON.",
                        seed=1200 + trial + attempt * 100,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                    )
                    record = {
                        "prompt": question,
                        "actual_query": query,
                        "response": asdict(response),
                    }
                    write_json(directory / f"raw_{trial}_{attempt}.json", record)
                    try:
                        answer = _parse_response_content(response.text)
                        record["answer"] = answer
                        record["integral"] = integrate_construction(
                            points,
                            owners,
                            pitch,
                            answer["regions"],
                            densities,
                            thickness_bounds=checked_bounds,
                        )
                        write_json(path, record)
                        error = None
                        break
                    except (ValueError, KeyError, TypeError) as exc:
                        error = exc
                        record["validation_error"] = str(exc)
                        write_json(directory / f"raw_{trial}_{attempt}.json", record)
                if error is not None:
                    raise ValueError(
                        f"{scene} round {trial} invalid after schema repair: {error}"
                    )
            previous = answer
            integral = integrate_construction(
                points,
                owners,
                pitch,
                answer["regions"],
                densities,
                thickness_bounds=checked_bounds,
            )
            results.append(integral["moments"]["mass_kg"])
            print(scene, trial, results[-1], flush=True)
        predictions.append(
            {
                "scene_id": scene,
                "mass_kg": float(np.mean(results)),
                "round_masses_kg": results,
            }
        )
    write_json(args.output / "predictions.json", predictions)
    write_json(
        args.output / "prediction_seal.json",
        {"predictions.json": file_sha256(args.output / "predictions.json")},
    )
    weights = json.loads((args.root / "filtered_product_weights.json").read_text())
    target = [float(weights.get(s, weights.get(s.split("_")[0]))) for s in ids]
    report = {
        "metrics": mass_metrics(target, [p["mass_kg"] for p in predictions]),
        "scope": "five exposed development objects; reference geometry; VLM construction inference",
    }
    write_json(args.output / "report.json", report)
    return report


def reanalyse_library(args):
    """Paired parameter-source ablation with exactly the same saved VLM answers."""
    verify_prediction_seal(args.output)
    ids = json.loads(args.cohort.read_text(encoding="utf-8"))["object_ids"]
    predictions = []
    for scene in ids:
        source = args.geometry / "objects" / scene
        original = json.loads((source / "legacy_prediction.json").read_text())
        densities = {
            m["name"]: float(np.mean(m["density"])) for m in original["materials"]
        }
        thicknesses = {
            m["name"]: float(np.mean(m["thickness_m"])) for m in original["materials"]
        }
        with np.load(source / "nearest_surface_field.npz") as field:
            points, owners, pitch = (
                field["centers_m"],
                field["part_ids"],
                float(field["pitch_m"]),
            )
        masses = []
        for trial in range(3):
            answer = json.loads(
                (args.output / scene / f"round_{trial}.json").read_text()
            )["answer"]
            result = integrate_construction(
                points, owners, pitch, answer["regions"], densities, thicknesses
            )
            masses.append(result["moments"]["mass_kg"])
        predictions.append(
            {
                "scene_id": scene,
                "mass_kg": float(np.mean(masses)),
                "round_masses_kg": masses,
            }
        )
    write_json(args.output / "library_predictions.json", predictions)
    write_json(
        args.output / "library_prediction_seal.json",
        {
            "library_predictions.json": file_sha256(
                args.output / "library_predictions.json"
            )
        },
    )
    weights = json.loads((args.root / "filtered_product_weights.json").read_text())
    target = [float(weights.get(s, weights.get(s.split("_")[0]))) for s in ids]
    report = {
        "metrics": mass_metrics(target, [r["mass_kg"] for r in predictions]),
        "scope": "same VLM answers; positive shell thicknesses replaced by supplied material prior means",
        "additional_VLM_calls": 0,
    }
    write_json(args.output / "library_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "assets", "geometry", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--cohort", type=Path, default=Path("configs/five_object_pilot.json")
    )
    parser.add_argument("--surface-hints", action="store_true")
    parser.add_argument("--allow-unconstrained-thickness", action="store_true")
    parser.add_argument("--stage", choices=("run", "library"), default="run")
    args = parser.parse_args()
    print(
        json.dumps(run(args) if args.stage == "run" else reanalyse_library(args)),
        flush=True,
    )


if __name__ == "__main__":
    main()
