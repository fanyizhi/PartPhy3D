"""Render matched-view failure cases and replay a fixed-wall scale intervention."""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from audit_abo_validation_limits import ROOT, read, sealed

OUT = ROOT / "structure_cases"
IDS = ("B07H8VCM68_A1RKKUPIHCS9HS", "B07QHYD94M_ATVPDKIKX0DER")


def rescale(row, ratio):
    """Keep the shell gauge in metres as area changes, including the volume cap."""
    area_sum = sum(p["area_share"] for p in row["parts"])
    mass = 0.
    for p in row["parts"]:
        if p["construction"] == "shell":
            area = row["area_m2"] * p["area_share"] / area_sum * ratio**2
            vol = min(area * p["thickness_mm"] / 1000, p["outer_volume_m3"] * ratio**3)
        else:
            vol = p["occupied_volume_m3"] * ratio**3
        mass += vol * p["density_kg_m3"]
    return mass


def build(dataset, panels):
    audit = {r["scene"]: r for r in read(ROOT / "validation_failure_audit.json")["rows"]}
    cache = read(ROOT / "nerf_run/predictions.json")
    records = {}
    for batch in panels.glob("val_*"):
        for scene, r in sealed(batch / "panels").items():
            if scene not in IDS:
                continue
            a = audit[scene]
            assert math.isclose(rescale(r, 1.), a["physical"], rel_tol=1e-8)
            records[scene] = dict(a, parts=r["parts"], nerf_midpoint=sum(cache[scene])/2,
                fixed_wall_kg=rescale(r, a["catalog_length_m"]/a["predicted_length_m"]))
    OUT.mkdir(exist_ok=True)
    (OUT / "source.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(8.5, 4.4), layout="constrained")
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 2.3])
    for row, scene in enumerate(IDS):
        r = records[scene]
        for j, view in enumerate([0, 8, 20]):
            ax = fig.add_subplot(grid[row, j])
            with Image.open(dataset / "scenes" / scene / "images" / f"{scene}_{view:02d}.png") as im:
                rgba = im.convert("RGBA")
                ax.imshow(rgba.crop(rgba.getbbox()))
            if row == 0:
                ax.set_title(f"View {view:02d}")
            if j == 0:
                ax.set_ylabel(("Wardrobe", "Sofa")[row], fontsize=10, weight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        ax = fig.add_subplot(grid[row, 3])
        parts = r["parts"]
        yy = np.arange(len(parts))
        values = [p["mass_kg"] for p in parts]
        ax.barh(yy, values, color=["#926446" if p["name"]=="internal frame" else "#538086" for p in parts])
        ax.set_yticks(yy, [p["name"].capitalize() for p in parts])
        ax.invert_yaxis()
        ax.set_xlim(0, max(values)*1.28)
        for y,v in zip(yy,values):
            ax.annotate(f"{v:.2f}", (v,y), xytext=(3,0), textcoords="offset points", va="center")
        ax.set_xlabel("Assigned component mass (kg)")
        ax.spines[["top","right"]].set_visible(False)
    for ext in ("png","svg","pdf"):
        fig.savefig(OUT / f"case_plate.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("panels", type=Path)
    args = parser.parse_args()
    build(args.dataset, args.panels)

