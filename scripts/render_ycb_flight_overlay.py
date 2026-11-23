"""Draw full MuJoCo flight paths and ghosted poses of a scanned hammer."""
import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial import ConvexHull


def run(source,out):
    out.mkdir(parents=True,exist_ok=True)
    with np.load(source/"render_geometry.npz",allow_pickle=False) as data:
        vertices,faces,head,com=[data[k] for k in ("vertices","faces","head","centres")]
    with np.load(source/"tracks.npz",allow_pickle=False) as data:
        t,xyz,rotation=[data[k] for k in ("time_s","com_m","rotation")]
    hull=vertices[ConvexHull(vertices).vertices]
    floor_idx=[]
    geometry=[]
    for j in range(3):
        if np.min(hull[:,2]-com[j,2]+xyz[j,0,2])<=0:
            raise ValueError("The hammer starts on or below the floor")
        z=np.einsum("ti,vi->tv",rotation[j,:,2,:],hull-com[j])+xyz[j,:,2,None]
        hit=np.flatnonzero(z.min(1)<=0)
        if not len(hit):
            raise ValueError("Simulation does not reach the ground")
        floor_idx.append(int(hit[0]-1))
        geometry.append(np.einsum("tij,j->ti",rotation[j],com[0]-com[j])+xyz[j])
    camera=np.array([[.21,-.13,1.0],[1.,0.,-.21],[0.,1.,.13]])
    def project(p):
        return p@camera
    projected_com=project(xyz)
    projected_geom=project(np.array(geometry))
    summary=dict(protocol="Matched impulses; terminate each display at last airborne state",
        last_airborne_time_s=[float(t[k]) for k in floor_idx],
        max_com_geometric_separation_m=[float(np.linalg.norm(xyz[j,:k+1]-geometry[j][:k+1],axis=1).max())
                                        for j,k in enumerate(floor_idx)],
        last_airborne_com_m=[xyz[j,k].tolist() for j,k in enumerate(floor_idx)])
    s=json.loads((source/"result.json").read_bytes())
    (out/"flight.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    np.savez_compressed(out/"flight_tracks.npz",t_s=t,com_m=xyz,geometric_center_m=geometry,last_airborne_frame=floor_idx)
    fig,axes=plt.subplots(1,3,figsize=(12,4.25),facecolor="#f6f9fb")
    fig.subplots_adjust(left=.028,right=.976,bottom=.15,top=.77,wspace=.07)
    fig.text(.03,.935,"Which point follows the parabola?",fontsize=21,weight="semibold",color="#173345")
    fig.text(.031,.870,"YCB 048   /   MuJoCo flight   /   Identical launch impulses",
             fontsize=10,color="#527084")
    red,blue="#c33340","#007cb0"
    base_x=np.arange(-4,4.1,.24)
    base_y=np.arange(-.72,4.2,.24)
    corners=[]; shades=[]
    for i,x in enumerate(base_x):
        for j,y in enumerate(base_y):
            corners.append(project(np.array([[x,y,0],[x+.24,y,0],[x+.24,y+.24,0],[x,y+.24,0]]))[:,:2])
            shades.append("#9eb7c8" if (i+j)%2 else "#c2d3de")
    lines=[];object_layers=[];ghost_layers=[];states=[]
    physics_last=max(floor_idx)
    fig.legend(handles=[Line2D([],[],c=red,lw=2,label="Centre of mass"),
                        Line2D([],[],c=blue,lw=2,label="Geometric centre")],
               loc="lower right",bbox_to_anchor=(.973,.024),ncol=2,frameon=False,fontsize=9)
    clock=fig.text(.03,.055,"",fontsize=9,color="#496273")
    for j,ax in enumerate(axes):
        ax.set_facecolor("#dcecf6")
        ax.set(xlim=(-.23,3.55),ylim=(-.24,2.32),aspect="equal")
        ax.set_xticks([]);ax.set_yticks([])
        ax.spines[:].set_visible(False)
        ax.set_title(s["models"][j]["method"],loc="left",fontsize=10,pad=9,weight="semibold",color="#183343")
        ax.add_collection(PolyCollection(corners,facecolors=shades,edgecolors="none"))
        colors=np.where(head[:,None], np.array([.30,.40,.47]) if j!=2 else np.array([.68,.76,.80]),
                        np.array([.55,.34,.17]))
        if j==0:
            colors=np.full_like(colors,[.39,.44,.48])
        def surface(index):
            pose=(vertices-com[j])@rotation[j,index].T+xyz[j,index]
            rendered=project(pose)[faces]
            order=np.argsort(-rendered[:,:,2].mean(1))
            return rendered[order,:,:2],colors[order]
        states.append({k:surface(k) for k in range(0,floor_idx[j]+1,20)}|{floor_idx[j]:surface(floor_idx[j])})
        ghosts=[]
        for k in np.linspace(0,floor_idx[j],9).astype(int):
            tri,rgb=surface(k)
            ghost=PolyCollection(tri,facecolors=rgb,alpha=.16,linewidths=0,rasterized=True,zorder=2)
            ax.add_collection(ghost);ghost.set_visible(False);ghosts.append((k,ghost))
        ghost_layers.append(ghosts)
        moving=PolyCollection([],linewidths=0,rasterized=True,zorder=4)
        ax.add_collection(moving);object_layers.append(moving)
        a,=ax.plot([],[],c=red,lw=1.8,zorder=3)
        b,=ax.plot([],[],c=blue,lw=1.55,ls="--",zorder=3)
        d,=ax.plot([],[],"o",c=red,mec="white",ms=4,zorder=5)
        lines.append((a,b,d))
    def update(k):
        for j,(a,b,d) in enumerate(lines):
            last=min(k,floor_idx[j])
            a.set_data(*projected_com[j,:last+1,:2].T)
            b.set_data(*projected_geom[j,:last+1,:2].T)
            d.set_data([projected_com[j,last,0]],[projected_com[j,last,1]])
            for step,ghost in ghost_layers[j]:
                ghost.set_visible(step<last)
            tri,rgb=states[j][last]
            object_layers[j].set_verts(tri)
            object_layers[j].set_facecolor(rgb)
        clock.set_text(f"{t[k]:.2f} s   ·   Shared camera and scale   ·   Each panel stops just before contact")
    plt.rcParams.update({"animation.ffmpeg_path":imageio_ffmpeg.get_ffmpeg_exe(),"pdf.fonttype":42})
    with FFMpegWriter(fps=24,codec="libx264",extra_args=["-pix_fmt","yuv420p","-crf","17"]).saving(
            fig,str(out/"flight_overlay.mp4"),dpi=144) as writer:
        for k in list(range(0,physics_last,20))+[physics_last]:
            update(k)
            for _ in range(16 if k in (0,physics_last) else 2):
                writer.grab_frame()
    for fmt in ("png","pdf"):
        fig.savefig(out/f"flight_overlay.{fmt}",dpi=300)
    print(summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    run(args.source,args.out)
