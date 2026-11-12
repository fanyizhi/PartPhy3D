"""Run the reference study in four steps: prepare, infer, evaluate, figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "infer",
            "bind-native",
            "evaluate",
            "domain-evaluate",
            "domain-figures",
            "figures",
            "all",
        ],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/abo500_five"))
    parser.add_argument(
        "--scene", help="Scene ID for prepare; default is the historical chaise"
    )
    parser.add_argument("--figures", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"),
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--resume", action="store_true", help="Reuse already completed stages"
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("configs/evidence/chaise_surface_material_review.json"),
    )
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--construction", type=Path)
    parser.add_argument(
        "--surface-audit",
        action="store_true",
        help="Bind outer surface material independently of hidden structural role",
    )
    args = parser.parse_args()
    args.figures = args.figures or args.output / "figures"

    from tactile3d.experiments import reference_evaluation as evaluation
    from tactile3d.experiments import reference_inference as inference
    from tactile3d.experiments import reference_study as core
    from tactile3d.figures import reference_study as figures

    figures.configure_style()
    if args.stage == "bind-native":
        inference.infer_native_parts(args)
        return

    if args.stage in {"domain-evaluate", "domain-figures"}:
        if args.construction is None:
            parser.error(
                "domain-evaluate requires --construction with sealed VLM hypotheses"
            )
        if args.stage == "domain-evaluate":
            evaluation.evaluate_support_domains(args)
        figures.draw_domain_retention(args)
        return

    def step(operation, marker=None):
        if args.resume and marker is not None and (args.output / marker).is_file():
            print(f"Reuse: {marker}", flush=True)
        else:
            operation(args)

    if args.stage in {"prepare", "all"}:
        step(core.prepare, "input_manifest.json")
        step(inference.prepare_views, "views_contacts/view_selection.json")
    if args.stage in {"infer", "all"}:
        step(core.run_vlm, "vlm/seal.json")
        core.finalize_vlm(args)
        step(inference.infer_views, "views_contacts/view_count_results.json")
        step(inference.infer_color, "views_contacts/color_control/answers.json")
        step(inference.query_properties, "active_state.json")
    if args.answers is None:
        color = args.output / "views_contacts/color_control/answers.json"
        args.answers = color if color.is_file() else None
    if args.stage in {"evaluate", "all"}:
        step(core.benchmark, "part_state.json")
        step(evaluation.evaluate_matched_qem, "matched_qem.json")
        evaluation.evaluate_roles(args)
        evaluation.evaluate_contacts(args)
        evaluation.evaluate_vlm(args)
        evaluation.evaluate_grounding(args)
    if args.stage in {"figures", "all"}:
        figures.draw_properties(args)
        figures.draw_contacts(args)
        evaluation.evaluate_grounding(args)
        figures.finalize_assets(args)


if __name__ == "__main__":
    main()
