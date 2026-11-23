"""Render the original Objaverse flower mesh under an explicit beam wind test."""
import argparse
from pathlib import Path
import hashlib
import json

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import PolyCollection
import numpy as np
from scipy.integrate import solve_ivp
import trimesh


def run(asset, out):
    out.mkdir(parents=True, exist_ok=True)
    scene = trimesh.load(asset, force="scene")
    # Normalize the asset's up-axis to the assumed experimental height in SI.
    height, radius, elastic, density = .45, .0025, 90e6, 650.
    air, drag, damping = 1.225, 1.2, .15
    width = .008
    base, top = scene.bounds[:,1]
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name]
        xyz = trimesh.transform_points(mesh.vertices, transform)[:,[0,2,1]]
        xyz[:,2] -= base
        xyz *= height/(top-base)
        face_uv = mesh.visual.uv[mesh.faces].mean(1) % 1
        texture = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"))
        iy = np.clip(((1-face_uv[:,1])*texture.shape[0]).astype(int),0,texture.shape[0]-1)
        ix = (face_uv[:,0]*texture.shape[1]).astype(int)
        rgb = texture[iy,ix]/255
        length = float(np.ptp(xyz[:,2]))
        u = (xyz[:,2]-xyz[:,2].min())/length
        mode = u*u*(6-4*u+u*u)/3  # static uniform-load cantilever shape, normalized at tip
        parts.append((xyz, mesh.faces, rgb, length, mode))
    t = np.linspace(0,2.6,66)
    wind = lambda s: 1.0*np.sin(np.pi*np.clip((s-.2)/1.2,0,1))**2
    I = np.pi*radius**4/4
    states = []
    for xyz,faces,rgb,length,mode in parts:
        omega = 1.875104**2*np.sqrt(elastic*I/(density*np.pi*radius**2*length**4))
        def ode(time, y):
            load = .5*air*drag*width*wind(time)**2
            tip = load*length**4/(8*elastic*I)
            return [y[1],omega**2*(tip-y[0])-2*damping*omega*y[1]]
        states.append(solve_ivp(ode,(0,t[-1]),[0,0],t_eval=t,rtol=1e-8,atol=1e-10).y[0])
    tallest = int(np.argmax([p[3] for p in parts]))
    amplitude = states[tallest]*1000
    out.joinpath("result.json").write_text(json.dumps(dict(asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        asset_id="18f5fba66f944aad9ae84ea2172b9b78",height_m=height,beam_radius_m=radius,youngs_pa=elastic,
        density_kg_m3=density,drag_width_m=width,peak_wind_m_s=1.,longest_beam_peak_mm=float(amplitude.max()),
        model="conditional damped Euler-Bernoulli beam approximation"),indent=2),encoding="utf-8")
    np.savez_compressed(out/"curves.npz",time_s=t,wind_m_s=wind(t),tip_m=states)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(8,4.1),gridspec_kw={"width_ratios":[1.15,1]},layout="constrained")
    fig.suptitle("Lavender under a prescribed gust",fontsize=16,weight="semibold")
    ax.set_title("a   Objaverse surface",loc="left",fontsize=10)
    ax.set(xlim=(-.20,.26),ylim=(-.02,.48),aspect="equal")
    ax.axis("off")
    right=np.array([.976,.217]);right/=np.linalg.norm(right)
    surface=PolyCollection([],linewidth=0,antialiased=False,rasterized=True)
    ax.add_collection(surface)
    bx.set_title("b   Longest stem response",loc="left",fontsize=10)
    bx.plot(t,amplitude,color="#634392",lw=2,label="Tip displacement")
    bx.set(xlabel="Time (s)",ylabel="Tip displacement (mm)")
    wx=bx.twinx()
    wx.plot(t,wind(t),color="#698885",ls=":",lw=1.5)
    wx.set_ylabel("Prescribed wind (m/s)",color="#698885")
    for a in (bx,wx): a.spines["top"].set_visible(False)
    cursor=bx.axvline(0,lw=.8,color=".6")
    caption=ax.text(.5,.025,"",transform=ax.transAxes,ha="center",fontsize=8)
    plt.rcParams.update({"animation.ffmpeg_path":imageio_ffmpeg.get_ffmpeg_exe(),"pdf.fonttype":42})
    with FFMpegWriter(fps=24,codec="libx264",extra_args=["-pix_fmt","yuv420p"]).saving(fig,str(out/"lavender_wind.mp4"),dpi=144) as writer:
        for i in range(len(t)):
            tris, depths, colors = [], [], []
            for (xyz,faces,rgb,length,mode), displacements in zip(parts,states):
                moving=xyz.copy()
                moving[:,:2]+=right*mode[:,None]*displacements[i]
                tri=moving[faces]
                tris.append(np.stack([tri[:,:,:2]@right,tri[:,:,2]],axis=-1))
                depths.append((tri[:,:,:2]@np.array([-right[1],right[0]])).mean(1))
                colors.append(rgb)
            order=np.argsort(np.concatenate(depths))
            surface.set_verts(np.concatenate(tris)[order])
            surface.set_facecolor(np.concatenate(colors)[order])
            cursor.set_xdata([t[i],t[i]])
            caption.set_text(f"{t[i]:.2f} s   ·   1× displacement")
            writer.grab_frame()
            if i==int(np.argmax(amplitude)):
                fig.savefig(out/"lavender_wind.png",dpi=250)
                fig.savefig(out/"lavender_wind.pdf",dpi=250)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("asset",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    run(a.asset,a.output)
