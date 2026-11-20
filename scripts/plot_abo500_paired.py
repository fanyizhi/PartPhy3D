"""Render the four paired mass metrics from the sealed benchmark review."""
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from evaluate_particle_moments import OUTPUT


def main():
    root = OUTPUT.parent
    report = json.loads((root/"paired_review.json").read_bytes())["splits"]
    metrics = [("ade_kg", "ADE (kg)", -1), ("alde", "ALDE", -1),
               ("ape", "APE (pp)", -100), ("mnre", "MnRE", 1)]
    refs = [("image", "Image prior", "#855f81", "s", -.13),
            ("nerf", "NeRF2Physics", "#186d66", "o", .13)]
    splits = ["train","val","test","all"]
    plt.rcParams.update({"font.size":8,"pdf.fonttype":42,"svg.fonttype":"none"})
    fig, axes = plt.subplots(1,4,figsize=(7.15,2.65),sharey=True)
    fig.subplots_adjust(left=.18,right=.985,top=.76,bottom=.24,wspace=.28)
    for ax,(metric,title,scale) in zip(axes,metrics):
        ax.axvline(0, lw=.8,color="#848990",zorder=0)
        ax.axhline(2.5,lw=.6,color=".8")
        for j,split in enumerate(splits):
            for ref,_,color,marker,dy in refs:
                stat = report[split]["paired_ours_minus"][ref][metric]
                lo,hi = sorted(np.asarray(stat["ci95"])*scale)
                x = stat["mean"]*scale
                ax.errorbar(x,j+dy,xerr=[[x-lo],[hi-x]],fmt=marker,ms=4,lw=1.15,
                            capsize=2.3,color=color,mec=color)
        ax.set_title(title, loc="left",fontsize=9,fontweight="semibold")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis="both",length=2.5,labelsize=7.5)
        ax.tick_params(axis="y",length=0)
        ax.spines[["left","top","right"]].set_visible(False)
    axes[0].set_yticks(range(4),["Train · 300","Validation · 100","Test · 100","All · 500"])
    axes[0].set_ylim(3.5,-.5)
    legend = [Line2D([],[],marker=m,color=c,lw=1,label=f"Ours vs. {label}") for _,label,c,m,_ in refs]
    fig.legend(handles=legend,loc="upper center",ncol=2,frameon=False,fontsize=8)
    fig.supxlabel("Paired improvement over comparator  (positive favours Ours)",fontsize=8,y=.025)
    for ext in ("pdf","svg","png"):
        fig.savefig(root/f"paired_mass.{ext}",dpi=350)
    plt.close(fig)


if __name__=="__main__":
    main()
