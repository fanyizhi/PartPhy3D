"""Replot published force and DIC displacement on beech LVL columns."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl


def main(data, out):
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"pdf.fonttype":42,"svg.fonttype":"none","font.size":9})
    fig,axes=plt.subplots(1,3,figsize=(8.6,3.1),sharex=True,sharey=True,layout="constrained")
    source={}
    rows=[]
    for ax, name in zip(axes,["S03","S04","S05"]):
        book=openpyxl.load_workbook(data/f"{name}.xlsx",read_only=True,data_only=True)
        s=book["Aramis"]
        pairs=np.array([(u,f) for f,u,*_ in s.iter_rows(min_row=3,values_only=True)
                        if isinstance(f,(int,float)) and isinstance(u,(int,float))])
        book.close()
        peak=int(pairs[:,1].argmax())
        rows.extend((name,i,*p) for i,p in enumerate(pairs))
        source[name]=dict(points=len(pairs),peak_force_kN=float(pairs[peak,1]),
                          displacement_at_peak_mm=float(pairs[peak,0]))
        ax.plot(*pairs[:peak+1].T,color="#207c7c",lw=1.7,label="Through peak")
        ax.plot(*pairs[peak:].T,color="#ad664d",lw=1.55,label="Post-peak")
        ax.scatter(*pairs[peak],c="#293943",s=14,zorder=3)
        ax.set(title=name,xlabel="Midspan displacement (mm)")
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Measured axial load (kN)")
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Measured load–displacement  |  Beech LVL columns",fontsize=13,weight="semibold")
    for fmt in ("png","pdf","svg"):fig.savefig(out/f"measured_buckling.{fmt}",dpi=300)
    (out/"result.json").write_text(json.dumps(source,indent=2),encoding="utf-8")
    with (out/"curves.csv").open("w",encoding="utf-8") as f:
        f.write("specimen,sample,midspan_mm,measured_load_kN\n")
        for row in rows: f.write(",".join(map(str,row))+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    main(args.data,args.out)
