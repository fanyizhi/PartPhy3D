"""Replay role-aware shell constraints and bounded mass fusion on frozen inputs."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluate_abo50_mass_hypotheses import mass_metrics, paired_comparison

BASE = Path(__file__).resolve().parents[1] / "paper_assets/abo100_extension_20261120"
OUT = BASE / "role_replay"
KEYS = ("ade_kg", "alde", "ape", "mnre")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def bounded(image, physical):
    """Equal log weights near agreement, bounded physical influence at large disagreement."""
    scale = math.log(2)
    return image * math.exp(0.5 * scale * math.tanh(math.log(physical / image) / scale))


def predict(row, evidence):
    image = row["image_prior_kg"]
    physical = row["construction_mass_kg"]
    if physical is None:
        return dict(updated=image, role_only=image, bounded_only=image, edits=[])
    edits = []
    lower = physical
    if evidence:
        parts = evidence["parts"]
        if not math.isclose(sum(p["mass_kg"] for p in parts), physical, abs_tol=1e-6):
            raise ValueError("Part masses differ from selected physical branch")
        shares = sum(p["area_share"] for p in parts)
        for part in parts:
            if not (part["construction"] == "solid" and part["material"] in {"fabric", "leather"}
                    and re.search(r"\b(cover|upholstery|lining|slipcover)\b", part["name"], re.I)):
                continue
            # For unresolved layers use the 2 mm thin-cover reference as a hypothesis.
            thickness = part["thickness_mm"] if part["thickness_mm"] is not None else 2.0
            if not math.isfinite(thickness) or thickness <= 0 or shares <= 0:
                raise ValueError("Invalid shell dimensions")
            area = evidence["area_m2"] * part["area_share"] / shares
            volume = min(part["outer_volume_m3"], area * thickness / 1000)
            shell = volume * part["density_kg_m3"]
            lower += shell - part["mass_kg"]
            edits.append(dict(part=part["name"], thickness_mm=thickness,
                              thickness_assumed=part["thickness_mm"] is None,
                              old_kg=part["mass_kg"], shell_kg=shell))
    # Remaining inner material is unidentified. Project the image estimate onto
    # the feasible interval [known mass, infinity), conditional on the other parts.
    corrected = max(image, lower) if edits else physical
    return dict(updated=bounded(image, corrected), role_only=math.sqrt(image * corrected),
                bounded_only=bounded(image, physical), edits=edits,
                physical_after_projection_kg=corrected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, help="ABO scenes directory for RGB case panels")
    args = parser.parse_args()
    source = BASE / "final100/selected/predictions.json"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != read(source.with_name("prediction_seal.json"))["sha256"]:
        raise ValueError("Input seal mismatch")
    evidence = read(OUT / "component_evidence.json")["scenes"]
    rows = read(source)
    ids = [row["scene_id"] for row in rows]
    if ids != read(BASE / "cohort100.json")["object_ids"] or len(set(ids)) != 100:
        raise ValueError("Incomplete cohort")
    predictions = {r["scene_id"]: predict(r, evidence.get(r["scene_id"])) for r in rows}
    path = OUT / "predictions.json"
    path.write_bytes((json.dumps(predictions, indent=2) + "\n").encode("utf-8"))
    # Predictions are persisted before reading any evaluation label.
    score = read(BASE / "final100/score.json")
    if [r["scene_id"] for r in score["per_scene"]] != ids or score["prediction_sha256"] != digest:
        raise ValueError("Score order or provenance mismatch")
    actual = np.array([r["target_mass_kg"] for r in score["per_scene"]])
    methods = {key: np.array([p[key] for p in predictions.values()])
               for key in ("updated", "role_only", "bounded_only")}
    for key in ["mass_kg", "image_prior_kg", "nerf_mass_kg"]:
        methods[key] = np.array([r[key] for r in score["per_scene"]])
    stats = {}
    for name, mask in {"all": np.ones(100, bool),
                       **{b: np.array([r["batch"] == b for r in rows])
                          for b in ("prior_50", "additional_50")}}.items():
        stats[name] = {key: mass_metrics(actual[mask], v[mask]) for key,v in methods.items()}
    stats["paired"] = {key: paired_comparison(list(zip(actual, methods["updated"], methods[key])))
                       for key in ["mass_kg", "image_prior_kg", "nerf_mass_kg", "bounded_only"]}
    stats["prediction_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "metrics.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    plot(actual, methods, stats["all"])
    if args.scenes:
        plot_covers(args.scenes, predictions, score["per_scene"])


def plot_covers(scenes, predictions, records):
    cases = [r for r in records if predictions[r["scene_id"]]["edits"]]
    fig, axes = plt.subplots(len(cases), 2, figsize=(7, 2.4 * len(cases)),
                             squeeze=False, layout="constrained")
    for (photo, bars), row in zip(axes, cases):
        scene = row["scene_id"]
        pred = predictions[scene]
        photo.imshow(plt.imread(scenes / scene / "images" / f"{scene}_08.png"))
        photo.axis("off")
        photo.set_title(scene.split("_")[0] + "  /  view 08")
        values = [row["mass_kg"], pred["role_only"], pred["updated"], row["target_mass_kg"]]
        bars.barh(range(4), values, color=["#888888","#74A8C6","#5A76A8","#222222"])
        bars.set_yticks(range(4), ["Frozen", "Role only", "Combined", "Catalogue"])
        bars.invert_yaxis()
        bars.set(xlim=(0, 170), xlabel="Object mass (kg)")
        for y, value in enumerate(values):
            bars.text(value + 2, y, f"{value:.1f}", va="center")
        bars.spines[["top", "right"]].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"cover_cases.{ext}", dpi=220)
    plt.close(fig)


def plot(actual, methods, metrics):
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
    for ax, key, label in zip(axes, ("mass_kg", "updated"), ("Frozen method", "Role constraint + bounded fusion")):
        ax.loglog(actual, methods[key], "o", ms=4, color="#5A76A8", alpha=.65)
        ax.plot([.01, 300], [.01, 300], "--", color=".4", lw=1)
        ax.set(xlim=(.009, 330), ylim=(.009, 330), aspect="equal", xlabel="Catalogue mass (kg)",
               ylabel="Predicted mass (kg)", title=label)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"mass_agreement.{ext}", dpi=220)
    plt.close(fig)
    # Published aggregates have different protocols. Hollow markers distinguish them.
    methods_list = ["Ours (replay)", "NeRF2Physics (paired)", "PUGS", "PhysVGGT", "GS2Physics", "Gen2Physics"]
    matrix = [[metrics[k][m] * (100 if m == "ape" else 1) for m in KEYS] for k in ("updated","nerf_mass_kg")]
    matrix += [[9.461,.661,76.7,.576],[4.730,.574,78.7,.655],[9.884,.720,88.8,.559],[8.447,.651,117,.600]]
    fig, axes = plt.subplots(1,4, figsize=(9,3), sharey=True, layout="constrained")
    for j,ax in enumerate(axes):
        for i,vals in enumerate(matrix):
            ax.scatter(vals[j],i,facecolor="#5A76A8" if i<2 else "white",edgecolor="#354E73")
        ax.axhline(1.5,lw=.6,color=".7")
        ax.set_xlabel(["ADE (kg) ↓","ALDE ↓","APE (%) ↓","MnRE ↑"][j])
        ax.set_yticks(range(6),methods_list)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.supxlabel("Filled: same-object scores     Open: published aggregates", fontsize=9)
    for ext in ("pdf","png"):
        fig.savefig(OUT / f"five_baselines.{ext}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
