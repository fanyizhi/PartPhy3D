"""Material assignments and contact-specific priors on matched ABO camera views."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,LogNorm,ListedColormap
from matplotlib.patches import Patch
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"paper_assets/eccentric_sphere"
priors=yaml.safe_load((ROOT/"configs/material_priors.yaml").read_text())["materials"]
records=json.loads((ROOT/"paper_assets/chaise_reference_20261111_final/material_priors_raw.json").read_bytes())
materials=["fabric","wood","steel"]
friction={}
for name in materials:
    estimates=[m["friction_coefficient"] for row in records if row.get("valid")
               for m in row["answer"]["materials"] if m["name"]==name and m.get("friction_coefficient") is not None]
    if estimates: friction[name]=np.median(estimates)
colors=["#4687ae","#c48951","#826595"]
fig=plt.figure(figsize=(8.6,4.45),layout="constrained")
grid=fig.add_gridspec(3,5,height_ratios=[1,1,.065])
maps=[ListedColormap(colors),plt.get_cmap("viridis"),plt.get_cmap("magma"),plt.get_cmap("cividis")]
norms=[Normalize(.5,3.5),Normalize(0,1),LogNorm(1e8,2e11),LogNorm(300,8000)]
titles=["Input RGB","Assigned material","Static friction μ","Young’s E (Pa)","Density (kg/m³)"]
for row,p in enumerate(sorted((OUT/"inputs").glob("*.npz"))):
    data=np.load(p)
    rgba=data["rgba"]/255
    mask=rgba[:,:,3]>.5
    yy,xx=np.where(mask)
    crop=np.s_[max(0,min(yy)-6):max(yy)+7,max(0,min(xx)-6):max(xx)+7]
    pred=data["prediction"]
    layers=[pred,np.array([np.nan]+[friction.get(m,np.nan) for m in materials])[pred],
            data["modulus_pa"],np.array([np.nan]+[priors[m]["density_kg_m3"]["mean"] for m in materials])[pred]]
    for j,title in enumerate(titles):
        ax=fig.add_subplot(grid[row,j])
        if row==0: ax.set_title(title,fontsize=10)
        if j==0:
            ax.imshow((rgba[:,:,:3]*rgba[:,:,3:]+1-rgba[:,:,3:])[crop])
            ax.set_ylabel(["Upholstered chair","Counter stool"][row],fontsize=9)
        else:
            ax.imshow(np.where(mask[:,:,None],.79,1.) .repeat(3,axis=2)[crop])
            im=ax.imshow(np.where(mask & (pred>0),layers[j-1],np.nan)[crop],cmap=maps[j-1],
                         norm=norms[j-1],interpolation="nearest")
            if row==1 and j>=2:
                cb=fig.colorbar(im,cax=fig.add_subplot(grid[2,j]),orientation="horizontal")
                cb.ax.tick_params(labelsize=7)
        ax.set_xticks([]);ax.set_yticks([])
        ax.spines[:].set_visible(False)
leg=fig.add_subplot(grid[2,:2]);leg.axis("off")
leg.legend(handles=[Patch(color=c,label=n.title()) for c,n in zip(colors+[".79"],materials+["unknown"])],
           loc="center",ncol=2,fontsize=7,frameon=False)
for ext in ("png","pdf","svg"): fig.savefig(OUT/f"surface_priors.{ext}",dpi=320)
