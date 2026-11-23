"""Estimate and render the first transverse membrane mode of an ABO canopy."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import PolyCollection
import numpy as np
from scipy.integrate import solve_ivp
from PIL import Image
import trimesh
import yaml

SCENE = "B07YLYRWJ7_ATVPDKIKX0DER"


def run(batch, priors, out):
    out.mkdir(parents=True, exist_ok=True)
    proposal = next(r for r in json.loads((batch/"panels/predictions.json").read_bytes())
                    if r["scene_id"] == SCENE)
    mesh_path = batch/"reconstructions_sparse"/SCENE/"camera_space.ply"
    digest = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    if digest != proposal["mesh_sha256_before_winding_repair"]:
        raise ValueError("Canopy prediction and reconstruction are from different runs")
    cloth = next(p for p in proposal["open_frame_groups"] if p["kind"]=="sheet")
    width, length, _ = np.asarray(cloth["dimensions"])/1000
    thickness = cloth["thickness_mm"]/1000
    prior = yaml.safe_load(priors.read_text(encoding="utf-8"))["materials"]["fabric"]
    youngs = float(prior["youngs_modulus_pa"]["mean"])
    rho = prior["density_kg_m3"]["mean"]
    prestrain, damping, peak_pa = .001, .16, 12.
    tension = youngs*thickness*prestrain
    frequency = np.pi*np.sqrt(tension/(rho*thickness)*(width**-2+length**-2))
    def pressure(t):
        phase = np.clip((t-.20)/1.45,0,1)
        return peak_pa*np.sin(np.pi*phase)**2
    times = np.linspace(0,3.4,103)
    def ode(t,y):
        return [y[1], (16/np.pi**2)*pressure(t)/(rho*thickness)
                -2*damping*frequency*y[1]-frequency**2*y[0]]
    motion=solve_ivp(ode,(0,times[-1]),[0,0],t_eval=times,rtol=1e-9,atol=1e-11).y[0]
    mesh = trimesh.load_mesh(mesh_path,process=False)
    raw = np.asarray(mesh.vertices)
    rgb = mesh.visual.vertex_colors[:,:3]/255
    blue = (rgb[:,2]-rgb[:,0]>.08)&(rgb[:,1]-rgb[:,0]>.08)
    if blue.sum()<100:
        raise ValueError("Failed to localize blue material on mesh")
    # Use color only to localize the blue canopy; the frozen model supplies its material.
    xyz = (raw-mesh.bounds[0])*(proposal["predicted_max_length_m"]/mesh.extents.max())
    xyz[:,:2]-=xyz[blue,:2].mean(0)
    lo,hi=np.quantile(xyz[blue,:2],[.005,.995],axis=0)
    uv=np.clip((xyz[:,:2]-lo)/(hi-lo),0,1)
    mode=np.prod(np.sin(np.pi*uv),axis=1)*blue
    mode[~blue]=0
    if xyz[blue,2].mean() <= xyz[~blue,2].mean():
        raise ValueError("Roof vertices must be above the frame")
    summary=dict(scene=SCENE,mesh_sha256=digest,blue_vertices=int(blue.sum()),sheet=cloth,
                 density_kg_m3=rho,youngs_pa=youngs,pretension_n_per_m=tension,
                 assumed_prestrain=prestrain,assumed_damping_ratio=damping,
                 peak_net_pressure_pa=peak_pa,natural_frequency_hz=frequency/(2*np.pi),
                 peak_modal_displacement_mm=float(motion.max()*1000))
    (out/"result.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    np.savez_compressed(out/"curves.npz",time_s=times,modal_mm=motion*1000,
                        pressure_pa=pressure(times),roof_mode=mode,roof_selector=blue)
    image=batch/"observations/scenes"/SCENE/"images"/f"{SCENE}_00.png"
    with Image.open(image) as im:
        im=im.convert("RGBA")
        photo=np.asarray(im.crop(im.getchannel("A").getbbox()))
    right=np.array([.77,.638,0]);right/=np.linalg.norm(right)
    up=np.array([-right[1]*.44,right[0]*.44,np.sqrt(1-.44**2)])
    camera=np.stack([right,up,np.cross(right,up)],axis=1)
    baseline=xyz@camera
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42,"svg.fonttype":"none",
                        "animation.ffmpeg_path":imageio_ffmpeg.get_ffmpeg_exe()})
    fig=plt.figure(figsize=(10,4.6),facecolor="#fafcfc")
    grid=fig.add_gridspec(2,3,width_ratios=[.8,1.35,1],left=.038,right=.97,
                          bottom=.155,top=.78,wspace=.31,hspace=.26)
    input_ax=fig.add_subplot(grid[:,0])
    body=fig.add_subplot(grid[:,1])
    load=fig.add_subplot(grid[0,2])
    trace=fig.add_subplot(grid[1,2],sharex=load)
    fig.text(.04,.925,"Canopy response to a wind gust",fontsize=20,weight="semibold",color="#20343c")
    fig.text(.041,.862,"ABO-500  /  B07YLYRWJ7  /  fixed-frame membrane",color="#567078",fontsize=10)
    input_ax.imshow(photo)
    input_ax.set_title("a   Dataset input",loc="left")
    input_ax.axis("off")
    body.set_title("b   Reconstructed canopy",loc="left")
    body.set(xlim=(baseline[:,0].min()-.10,baseline[:,0].max()+.10),
             ylim=(baseline[:,1].min()-.10,baseline[:,1].max()+.25),aspect="equal")
    body.axis("off")
    surface=PolyCollection([],linewidths=0,antialiased=False,rasterized=True)
    body.add_collection(surface)
    body.annotate("Net roof pressure",xy=(.75,.82),xytext=(.75,.96),xycoords="axes fraction",
                  ha="center",color="#865844",fontsize=8,
                  arrowprops={"arrowstyle":"->","color":"#865844"})
    load.plot(times,pressure(times),color="#a86448",lw=2)
    trace.plot(times,motion*1000,color="#217d81",lw=2)
    for ax,key in [(load,"Pressure (Pa)"),(trace,"Center deflection (mm)")]:
        ax.spines[["top","right"]].set_visible(False)
        ax.set_ylabel(key)
        ax.grid(axis="y",color="#e2ebec",lw=.6)
    load.set_title("c   Load and response",loc="left")
    load.tick_params(labelbottom=False)
    trace.set_xlabel("Time (s)")
    markers=[ax.axvline(0,color="#90a1a6",lw=.8) for ax in (load,trace)]
    footer=fig.text(.04,.057,"",fontsize=9,color="#39555d")
    face_rgb=rgb[mesh.faces].mean(1)
    peak=int(np.argmax(motion))
    with FFMpegWriter(fps=24,codec="libx264",extra_args=["-pix_fmt","yuv420p"]).saving(
        fig,str(out/"canopy_wind.mp4"),dpi=144) as writer:
        for i,t in enumerate(times):
            state=xyz.copy()
            state[:,2]-=motion[i]*mode
            triangles=(state@camera)[mesh.faces]
            order=np.argsort(triangles[:,:,2].mean(1))
            surface.set_verts(triangles[order,:,:2])
            surface.set_facecolor(face_rgb[order])
            for mark in markers:
                mark.set_xdata([t,t])
            footer.set_text(f"{t:0.2f} s   ·   Roof center deflection {motion[i]*1000:+.1f} mm   ·   1× displacement")
            writer.grab_frame()
            if i==peak:
                for fmt in ("pdf","svg","png"):
                    fig.savefig(out/f"canopy_wind.{fmt}",dpi=280)
    plt.close(fig)
    print(summary,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("batch","priors","output"):
        parser.add_argument(key,type=Path)
    args=parser.parse_args()
    run(args.batch,args.priors,args.output)

