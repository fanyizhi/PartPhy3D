"""Render simulated snapshots using the archived heterogeneous sofa particle input."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import trimesh

from tactile3d.physics.mpm_validation import run_mpm


def render(folder):
    points = dict(np.load(folder/"sofa_mpm_input.npz", allow_pickle=False))
    x = points["position_m"]
    np.testing.assert_allclose(points["mass_kg"], points["volume_m3"]*points["density_kg_m3"])
    states = [(0.,x)]
    def capture(step, time, pos):
        if step % 4 == 0:
            states.append((time,pos))
    result = run_mpm(points, dx=0.12307692307692308, steps=120, strain_rate=10, on_step=capture)
    reference = np.load(folder/"sofa_mpm_response.npz")["position_m"]
    np.testing.assert_allclose(states[-1][1], reference, rtol=0, atol=1e-10)
    mesh = trimesh.load_mesh(folder/"corrected_camera_space.ply", process=False)
    # Orthographic camera, with the same fixed world bounds in all panels.
    right = np.array([.89,.456,0.]); right /= np.linalg.norm(right)
    up = np.array([-right[1]*.32,right[0]*.32,np.sqrt(1-.32**2)])
    camera = np.stack([right,up,np.cross(right,up)],axis=1)
    proj = x @ camera
    tri = (mesh.vertices @ camera)[mesh.faces]
    order = np.argsort(tri[:,:,2].mean(1))
    rgb = mesh.visual.vertex_colors[mesh.faces,:3].mean(1)/255
    fig, axes = plt.subplots(1,3,figsize=(10.2,3.3))
    fig.subplots_adjust(left=.02,right=.98,top=.78,bottom=.23,wspace=.05)
    axes[0].add_collection(PolyCollection(tri[order,:,:2],facecolors=rgb[order],edgecolors="face",
                                          linewidths=.12,rasterized=True))
    palette = {"fabric":"#4275a1","foam":"#e4a648","wood":"#913f32"}
    names = points["material_names"][points["material_id"]]
    # A world-space cutaway exposes interior points; it only changes rendering.
    keep = np.flatnonzero(x[:,1] >= np.median(x[:,1]))
    keep = keep[np.argsort(proj[keep,2])]
    axes[1].scatter(*proj[keep,:2].T,c=[palette[n] for n in names[keep]],s=6,lw=0)
    order = np.argsort(proj[:,2])
    max_mm = np.linalg.norm(states[-1][1]-x,axis=1).max()*1000
    dots = axes[2].scatter(*proj[order,:2].T,c=np.zeros(len(x)),s=5,lw=0,cmap="magma",
                         vmin=0,vmax=max_mm)
    for ax,title in zip(axes,["a   Reconstructed mesh","b   Interior material cutaway","c   MPM displacement"]):
        ax.set_title(title,fontsize=11,loc="left",weight="semibold")
        ax.set(xlim=(proj[:,0].min()-.1,proj[:,0].max()+.1),
               ylim=(proj[:,1].min()-.13,proj[:,1].max()+.13),aspect="equal")
        ax.axis("off")
    axes[1].legend(handles=[Line2D([],[],marker="o",lw=0,color=v,label=k.title()) for k,v in palette.items()],
                   loc="upper center",bbox_to_anchor=(.5,-.02),frameon=False,ncol=3,fontsize=8,
                   columnspacing=.5,handletextpad=.15)
    cb=fig.colorbar(dots,cax=fig.add_axes([.72,.17,.225,.022]),orientation="horizontal")
    cb.set_label("Displacement magnitude (mm)",fontsize=8)
    cb.ax.tick_params(labelsize=8)
    label=fig.text(.027,.16,"",fontsize=10)
    fig.text(.027,.086,"Free compression  ·  motion ×35",fontsize=8)
    frames=[]
    for t,pos in states:
        delta=pos-x
        dots.set_offsets(((x+35*delta) @ camera)[order,:2])
        dots.set_array(np.linalg.norm(delta,axis=1)[order]*1000)
        label.set_text(f"Simulation time   {t*1000:.3f} ms")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
    for ext in ("pdf","png"):
        fig.savefig(folder/f"sofa_dynamics.{ext}",dpi=300)
    frames[0].save(folder/"sofa_dynamics.gif",save_all=True,append_images=frames[1:],
                   duration=[100]*(len(frames)-1)+[900],loop=0,disposal=2)
    plt.close(fig)
    print(len(states),result["duration_s"],max_mm,result["max_p2g_mass_residual_kg"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    render(parser.parse_args().folder)
