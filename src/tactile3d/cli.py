from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from tactile3d.config import load_config
from tactile3d.eval.metrics import evaluate_summary
from tactile3d.uncertainty.monte_carlo import run_inference
from tactile3d.utils.io import ensure_dir, write_json


def make_demo_image(output: str | Path) -> None:
    output = Path(output)
    ensure_dir(output.parent)
    image = Image.new("RGBA", (256, 256), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((48, 70, 208, 188), radius=34, fill=(70, 92, 115, 255))
    draw.ellipse((85, 42, 170, 128), fill=(180, 188, 190, 255))
    draw.rectangle((82, 164, 116, 220), fill=(42, 42, 46, 255))
    draw.rectangle((148, 164, 182, 220), fill=(42, 42, 46, 255))
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="PartPhy3D research baseline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    infer = sub.add_parser("infer")
    infer.add_argument("--image", required=True)
    infer.add_argument("--config", default="configs/default.yaml")
    infer.add_argument("--output", required=True)

    eval_p = sub.add_parser("evaluate")
    eval_p.add_argument("--pred", required=True)
    eval_p.add_argument("--target", required=True)

    demo = sub.add_parser("make-demo-image")
    demo.add_argument("--output", required=True)

    abo_validate = sub.add_parser("abo500-validate", help="validate ABO-500 data and meshes")
    _add_abo_dataset_args(abo_validate)
    abo_validate.add_argument("--split", default="all")
    abo_validate.add_argument("--limit", type=int)

    abo_run = sub.add_parser("abo500-run", help="run mesh physical inference on ABO-500")
    _add_abo_dataset_args(abo_run)
    abo_run.add_argument("--config", default="configs/abo500_mesh.yaml")
    abo_run.add_argument("--split", default="test")
    abo_run.add_argument("--output", required=True)
    abo_run.add_argument("--calibration")
    abo_run.add_argument("--limit", type=int)
    abo_run.add_argument(
        "--vlm-backend",
        choices=("surface_only", "heuristic", "file", "openai_compatible"),
        help="override config VLM/interior backend",
    )
    abo_run.add_argument("--vlm-response-root", help="cached response root for file backend")
    abo_run.add_argument("--vlm-base-url", help="OpenAI-compatible VLM base URL")
    abo_run.add_argument("--vlm-model", help="served multimodal model name")

    abo_calibrate = sub.add_parser(
        "abo500-calibrate", help="fit weakly supervised material calibration from mass labels"
    )
    abo_calibrate.add_argument("--features", required=True)
    abo_calibrate.add_argument("--config", default="configs/abo500_mesh.yaml")
    abo_calibrate.add_argument("--output", required=True)

    abo_evaluate = sub.add_parser("abo500-evaluate", help="evaluate ABO-500 mass predictions")
    abo_evaluate.add_argument("--predictions", required=True)
    abo_evaluate.add_argument("--output")

    abo_demo = sub.add_parser("abo500-make-demo", help="create a tiny ABO-500-shaped fixture")
    abo_demo.add_argument("--root", required=True)

    mesh_run = sub.add_parser("mesh-run", help="infer physics from one metric textured mesh")
    mesh_run.add_argument("--mesh", required=True)
    mesh_run.add_argument("--mesh-unit-to-meter", required=True, type=float)
    mesh_run.add_argument("--config", default="configs/abo500_mesh_vlm.yaml")
    mesh_run.add_argument("--output", required=True)
    mesh_run.add_argument("--calibration")
    mesh_run.add_argument("--blender", default="blender")
    mesh_run.add_argument("--num-views", type=int, default=12)
    mesh_run.add_argument("--resolution", type=int, default=512)
    mesh_run.add_argument("--overwrite-renders", action="store_true")
    mesh_run.add_argument(
        "--vlm-backend",
        choices=("surface_only", "heuristic", "file", "openai_compatible"),
    )
    mesh_run.add_argument("--vlm-response-root")
    mesh_run.add_argument("--vlm-base-url")
    mesh_run.add_argument("--vlm-model")

    prepare_sft = sub.add_parser(
        "prepare-sft", help="build grouped multimodal SFT splits from expert annotations"
    )
    prepare_sft.add_argument("--annotations", required=True)
    prepare_sft.add_argument("--image-root")
    prepare_sft.add_argument("--output", required=True)
    prepare_sft.add_argument("--train-fraction", type=float, default=0.8)
    prepare_sft.add_argument("--val-fraction", type=float, default=0.1)
    prepare_sft.add_argument("--seed", type=int, default=17)
    prepare_sft.add_argument("--max-images", type=int, default=4)

    fit_psych = sub.add_parser(
        "fit-psychophysics", help="fit human perceptual scores from pairwise judgements"
    )
    fit_psych.add_argument("--comparisons", required=True)
    fit_psych.add_argument("--output", required=True)
    fit_psych.add_argument("--l2", type=float, default=1e-3)

    train_vlm = sub.add_parser("train-vlm", help="LoRA/QLoRA fine-tune a multimodal model")
    from tactile3d.training.train_vlm import add_training_arguments

    add_training_arguments(train_vlm)

    part_replace = sub.add_parser(
        "part-replace", help="replace a labelled mesh part and recompute physical properties"
    )
    part_replace.add_argument("--source", required=True)
    part_replace.add_argument("--face-labels", required=True)
    part_replace.add_argument("--target-part", required=True, type=int)
    part_replace.add_argument("--donor", required=True)
    part_replace.add_argument("--mesh-unit-to-meter", required=True, type=float)
    part_replace.add_argument("--source-density", required=True, type=float)
    part_replace.add_argument("--donor-density", required=True, type=float)
    part_replace.add_argument("--vertical-axis", type=int, default=2)
    part_replace.add_argument("--no-icp", action="store_true")
    part_replace.add_argument("--output", required=True)

    mesh_volume = sub.add_parser("mesh-volume", help="validate and integrate mesh volume")
    mesh_volume.add_argument("--mesh", required=True)
    mesh_volume.add_argument("--mesh-unit-to-meter", required=True, type=float)
    mesh_volume.add_argument(
        "--repair-strategy", choices=("none", "fill_only", "fill_then_hull"), default="fill_only"
    )
    mesh_volume.add_argument("--convex-hull-proxy", action="store_true")
    mesh_volume.add_argument("--density", type=float, help="optional uniform density in kg/m^3")
    mesh_volume.add_argument("--fill-fraction", type=float, default=1.0)
    mesh_volume.add_argument("--output")

    import_baseline = sub.add_parser(
        "import-mass-baseline", help="normalize PUGS or scalar baseline mass predictions"
    )
    import_baseline.add_argument("--predictions", required=True)
    import_baseline.add_argument("--targets", required=True)
    import_baseline.add_argument("--method", required=True)
    import_baseline.add_argument("--scene-id-separator", default="_")
    import_baseline.add_argument("--output", required=True)

    paper_figures = sub.add_parser(
        "paper-figures", help="compare prediction runs and export publication figures"
    )
    paper_figures.add_argument(
        "--run", action="append", required=True, metavar="METHOD=PREDICTIONS_JSON"
    )
    paper_figures.add_argument("--output", required=True)
    paper_figures.add_argument("--bootstrap-samples", type=int, default=2000)
    paper_figures.add_argument("--seed", type=int, default=17)
    paper_figures.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg", "pdf", "tiff"),
        default=("png", "svg", "pdf", "tiff"),
    )

    object_figures = sub.add_parser(
        "object-figures", help="render part-level physical property maps for one mesh"
    )
    object_figures.add_argument("--prediction", required=True)
    object_figures.add_argument("--mesh", required=True)
    object_figures.add_argument("--face-labels", required=True)
    object_figures.add_argument("--output", required=True)
    object_figures.add_argument("--max-faces", type=int, default=50000)
    object_figures.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg", "pdf", "tiff"),
        default=("png", "svg", "pdf", "tiff"),
    )

    abo_pilot = sub.add_parser(
        "abo-pilot", help="run the reproducible ABO sofa physics pilot and export paper figures"
    )
    abo_pilot.add_argument("--asset-dir", default="data/pilot/B075X4QMX3")
    abo_pilot.add_argument("--output", default="outputs/abo_sofa_pilot")
    abo_pilot.add_argument("--samples", type=int, default=8000)
    abo_pilot.add_argument("--seed", type=int, default=17)
    abo_pilot.add_argument("--no-download", action="store_true")
    abo_pilot.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg", "pdf", "tiff"),
        default=("png", "svg", "pdf"),
    )

    token_pilot = sub.add_parser(
        "token-mapping-pilot", help="ABO part-state and cross-LOD consistency pilot"
    )
    token_pilot.add_argument("--asset-dir", default="data/pilot/B075X4QMX3")
    token_pilot.add_argument("--output", default="outputs/part_token_pilot")
    token_pilot.add_argument("--samples", type=int, default=4000)
    token_pilot.add_argument("--seed", type=int, default=17)
    token_pilot.add_argument("--pitch", type=float, default=0.04)

    detail_pilot = sub.add_parser(
        "assembly-detail-pilot", help="inspect native sofa pieces and candidate leg connections"
    )
    detail_pilot.add_argument("--asset-dir", default="data/pilot/B075X4QMX3")
    detail_pilot.add_argument("--output", default="outputs/assembly_detail_pilot")

    args = parser.parse_args()
    if args.cmd == "infer":
        cfg = load_config(args.config)
        summary = run_inference(args.image, cfg, args.output)
        write_json(Path(args.output) / "summary.json", summary)
    elif args.cmd == "evaluate":
        metrics = evaluate_summary(args.pred, args.target)
        print(metrics)
    elif args.cmd == "make-demo-image":
        make_demo_image(args.output)
    elif args.cmd == "abo500-validate":
        from tactile3d.abo500.dataset import ABO500Dataset

        dataset = ABO500Dataset(args.root, args.mesh_root, args.mesh_unit_to_meter)
        print(json.dumps(dataset.validate(args.split, args.limit), indent=2))
    elif args.cmd == "abo500-run":
        from tactile3d.abo500.dataset import ABO500Dataset
        from tactile3d.abo500.pipeline import run_abo500_split

        cfg = load_config(args.config)
        _apply_vlm_overrides(cfg, args)
        unit_to_meter = args.mesh_unit_to_meter
        if unit_to_meter is None:
            unit_to_meter = cfg.get("dataset", {}).get("mesh_unit_to_meter")
        dataset = ABO500Dataset(args.root, args.mesh_root, unit_to_meter)
        result = run_abo500_split(
            dataset,
            cfg,
            args.split,
            args.output,
            calibration_path=args.calibration,
            limit=args.limit,
        )
        print(json.dumps(result, indent=2))
    elif args.cmd == "abo500-calibrate":
        from tactile3d.config import resolve_config_path
        from tactile3d.models.calibration import fit_mass_calibration
        from tactile3d.physics.materials import load_material_priors

        cfg = load_config(args.config)
        priors_path = resolve_config_path(
            cfg.get("materials", {}).get("priors", "configs/material_priors.yaml"),
            cfg.get("_config_dir"),
        )
        checkpoint = fit_mass_calibration(
            args.features,
            load_material_priors(priors_path),
            args.output,
            l2=float(cfg.get("calibration", {}).get("l2", 0.02)),
            max_iters=int(cfg.get("calibration", {}).get("max_iters", 500)),
        )
        print(json.dumps(checkpoint, indent=2))
    elif args.cmd == "abo500-evaluate":
        from tactile3d.eval.abo500 import evaluate_abo500_predictions

        result = evaluate_abo500_predictions(args.predictions, args.output)
        print(json.dumps(result["metrics"], indent=2))
    elif args.cmd == "abo500-make-demo":
        from tactile3d.abo500.demo import make_demo_dataset

        root, mesh_root = make_demo_dataset(args.root)
        print(json.dumps({"root": str(root), "mesh_root": str(mesh_root)}, indent=2))
    elif args.cmd == "mesh-run":
        from tactile3d.abo500.dataset import ABOScene, load_camera_views
        from tactile3d.abo500.pipeline import ABO500MeshPipeline
        from tactile3d.mesh.render_blender import render_mesh_views

        cfg = load_config(args.config)
        _apply_vlm_overrides(cfg, args)
        mesh_path = Path(args.mesh).expanduser().resolve()
        render_dir = Path(args.output).expanduser().resolve() / "renders"
        render_mesh_views(
            mesh_path,
            render_dir,
            blender_executable=args.blender,
            num_views=args.num_views,
            resolution=args.resolution,
            overwrite=args.overwrite_renders,
        )
        scene = ABOScene(
            scene_id=mesh_path.stem,
            scene_dir=render_dir,
            split="inference",
            mass_kg=None,
            mesh_path=mesh_path,
            mesh_unit_to_meter=args.mesh_unit_to_meter,
            views=tuple(load_camera_views(render_dir)),
        )
        prediction, _ = ABO500MeshPipeline(cfg, args.calibration).infer_scene(
            scene, Path(args.output) / "result"
        )
        print(json.dumps(prediction["object"], indent=2))
    elif args.cmd == "prepare-sft":
        from tactile3d.training.dataset import prepare_sft_dataset

        report = prepare_sft_dataset(
            args.annotations,
            args.output,
            image_root=args.image_root,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            seed=args.seed,
            max_images=args.max_images,
        )
        print(json.dumps(report, indent=2))
    elif args.cmd == "fit-psychophysics":
        from tactile3d.training.psychophysics import fit_bradley_terry

        result = fit_bradley_terry(args.comparisons, args.output, l2=args.l2)
        print(json.dumps(result, indent=2))
    elif args.cmd == "train-vlm":
        from tactile3d.training.train_vlm import run_training

        run_training(args)
    elif args.cmd == "part-replace":
        from tactile3d.mesh.editing import replace_mesh_part

        report = replace_mesh_part(
            args.source,
            args.face_labels,
            args.donor,
            args.target_part,
            args.output,
            mesh_unit_to_meter=args.mesh_unit_to_meter,
            source_density_kg_m3=args.source_density,
            donor_density_kg_m3=args.donor_density,
            use_icp=not args.no_icp,
            vertical_axis=args.vertical_axis,
        )
        print(json.dumps(report, indent=2))
    elif args.cmd == "mesh-volume":
        from tactile3d.mesh.volume import analyze_mesh_volume

        result = analyze_mesh_volume(
            args.mesh,
            args.mesh_unit_to_meter,
            repair_strategy=args.repair_strategy,
            convex_hull_proxy=args.convex_hull_proxy,
            density_kg_m3=args.density,
            fill_fraction=args.fill_fraction,
            output_path=args.output,
        )
        print(json.dumps(result, indent=2))
    elif args.cmd == "import-mass-baseline":
        from tactile3d.eval.import_baseline import import_mass_baseline

        report = import_mass_baseline(
            args.predictions,
            args.targets,
            args.output,
            method=args.method,
            scene_id_separator=args.scene_id_separator,
        )
        print(json.dumps(report, indent=2))
    elif args.cmd == "paper-figures":
        from tactile3d.figures.comparison import generate_comparison_figures

        report = generate_comparison_figures(
            _parse_runs(args.run),
            args.output,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            formats=tuple(args.formats),
        )
        print(json.dumps(report, indent=2))
    elif args.cmd == "object-figures":
        from tactile3d.figures.object_maps import generate_object_figures

        report = generate_object_figures(
            args.prediction,
            args.mesh,
            args.face_labels,
            args.output,
            formats=tuple(args.formats),
            max_faces=args.max_faces,
        )
        print(json.dumps(report, indent=2))
    elif args.cmd == "assembly-detail-pilot":
        from tactile3d.figures.assembly_pilot import run_assembly_pilot

        report = run_assembly_pilot(args.asset_dir, args.output)
        print(json.dumps({"output": args.output, "parts": len(report["parts"]),
                          "connections": report["connections"]}, indent=2))
    elif args.cmd == "token-mapping-pilot":
        from tactile3d.figures.token_pilot import run_token_pilot

        report = run_token_pilot(
            args.asset_dir, args.output, samples=args.samples, seed=args.seed, pitch=args.pitch
        )
        print(json.dumps({"output": args.output,
                          "mass": report["mass_prior_predictive_kg"],
                          "geometry": report["geometry"]}, indent=2))
    elif args.cmd == "abo-pilot":
        from tactile3d.figures.abo_pilot import generate_abo_sofa_pilot

        report = generate_abo_sofa_pilot(
            args.asset_dir,
            args.output,
            download_missing=not args.no_download,
            samples=args.samples,
            seed=args.seed,
            formats=tuple(args.formats),
        )
        print(json.dumps(report["posterior"], indent=2))


def _add_abo_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", required=True, help="ABO-500 root containing scenes/")
    parser.add_argument("--mesh-root", help="optional external directory containing scene meshes")
    parser.add_argument(
        "--mesh-unit-to-meter",
        type=float,
        help="metric length of one mesh unit; required unless each scene has mesh_scale.json",
    )


def _apply_vlm_overrides(cfg: dict, args: argparse.Namespace) -> None:
    vlm = cfg.setdefault("vlm", {})
    for argument, key in (
        ("vlm_backend", "backend"),
        ("vlm_response_root", "response_root"),
        ("vlm_base_url", "base_url"),
        ("vlm_model", "model"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            vlm[key] = value


def _parse_runs(values: list[str]) -> dict[str, str]:
    runs = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected METHOD=PATH, received {value!r}")
        method, path = value.split("=", 1)
        method = method.strip()
        if not method or not path.strip():
            raise ValueError(f"Expected non-empty METHOD=PATH, received {value!r}")
        if method in runs:
            raise ValueError(f"Duplicate method label: {method}")
        runs[method] = path.strip()
    return runs


if __name__ == "__main__":
    main()
