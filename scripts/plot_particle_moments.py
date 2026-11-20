"""Render cross-sections of exported material points with analytic moment errors."""
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from evaluate_particle_moments import OUTPUT, PARTICLES


def main():
    rows = json.loads(OUTPUT.read_bytes())
    labels = ["Solid", "Hollow", "Hollow (wall prior)", "Foam core", "Frame + foam"]
    colors = {"plastic": "#63788d", "foam": "#4a9187", "wood": "#bd8154"}
    plt.rcParams.update({"font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 4.35), layout="constrained")
    sub = fig.subfigures(2, 1, height_ratios=[1,1.55])
    axes = sub[0].subplots(1, 5)
    for ax, (key, r), label in zip(axes, rows.items(), labels):
        with np.load(PARTICLES/f"{key}_mpm_input.npz") as data:
            x, ids = data["position_m"]*1000, data["material_id"]
            for i, material in enumerate(data["material_names"]):
                use = (np.abs(x[:,2]) <= 8) & (ids==i)
                ax.scatter(x[use,0], x[use,1], s=3, color=colors[material], linewidths=0)
        ax.add_patch(Rectangle((-100,-100),200,200, fill=False, lw=.5, ec=".4"))
        ax.set(xlim=(-105,105), ylim=(-105,105), aspect="equal", title=label)
        ax.axis("off")
    handles = [Line2D([],[],marker="s",linestyle="none",color=c,label=m.title())
               for m,c in colors.items()]
    sub[0].legend(handles=handles, ncol=3, loc="lower center", frameon=False, fontsize=8,
                  bbox_to_anchor=(.5,-.08))
    ax1, ax2 = sub[1].subplots(1,2,sharey=True)
    yy = np.arange(len(rows))
    for ax,pair,title in [(ax1,"com_error_mm","Centre-of-mass error (mm)"),
                           (ax2,"inertia_relative","Inertia tensor error (%)")]:
        mul = 100 if pair=="inertia_relative" else 1
        for prefix, label, color, marker, offset in (
                ("uniform","Mass-matched uniform","#806b86","s",-.12),
                ("particle","Material points","#186d66","o",.12)):
            vals = [r[f"{prefix}_{pair}"] * mul for r in rows.values()]
            ax.scatter(vals, yy+offset, s=23, c=color, marker=marker, label=label)
        ax.set_xlabel(title)
        ax.grid(axis="x",color=".9",lw=.6)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
    ax1.set_yticks(yy,labels)
    ax1.set_ylim(4.6,-.6)
    ax2.legend(frameon=False, loc="upper right", fontsize=7)
    for ext in ("pdf","svg","png"):
        fig.savefig(OUTPUT.with_suffix("."+ext),dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
