"""Render the original Objaverse flower mesh under an paired MuJoCo beam comparison."""
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
import mujoco
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
    modulus_values = [elastic, 4*elastic]
    states, checks = [], []
    peak_load = .5*air*drag*width
    substeps = 40
    dt = (t[1]-t[0])/substeps
    for modulus in modulus_values:
        stems, validation = [], []
        for _, _, _, length, _ in parts:
            omega = 1.875104**2*np.sqrt(modulus*I/(density*np.pi*radius**2*length**4))
            compliance = length**4/(8*modulus*I)
            effective_mass = 1/(omega**2*compliance)
            stiffness = 1/compliance
            resistance = 2*damping*omega*effective_mass
            xml = f"""<mujoco><option timestep="{dt}" gravity="0 0 0" integrator="RK4"/>
              <worldbody><body><joint type="slide" axis="1 0 0" stiffness="{stiffness}" damping="{resistance}"/>
              <inertial pos="0 0 0" mass="{effective_mass}" diaginertia=".0001 .0001 .0001"/>
              </body></worldbody></mujoco>"""
            model=mujoco.MjModel.from_xml_string(xml)
            sim=mujoco.MjData(model)
            track=[0.]
            for k in range(1,len(t)):
                for _ in range(substeps):
                    sim.qfrc_applied[0]=peak_load*wind(sim.time+dt/2)**2
                    mujoco.mj_step(model,sim)
                track.append(float(sim.qpos[0]))
            def ode(time,y):
                return [y[1],omega**2*(compliance*peak_load*wind(time)**2-y[0])-2*damping*omega*y[1]]
            reference=solve_ivp(ode,(0,t[-1]),[0,0],t_eval=t,rtol=1e-9,atol=1e-11).y[0]
            residual=float(np.max(abs(reference-track)))
            if residual>1e-6:
                raise ValueError(f"MuJoCo modal response disagrees with the independent ODE: {residual}")
            stems.append(track)
            validation.append(dict(frequency_hz=float(omega/(2*np.pi)),max_residual_m=residual))
        states.append(stems)
        checks.append(validation)
    states=np.asarray(states)
    tallest=int(np.argmax([p[3] for p in parts]))
    amplitude=states[:,tallest]*1000
    stats=dict(asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        asset_id="18f5fba66f944aad9ae84ea2172b9b78",height_m=height,beam_radius_m=radius,
        youngs_pa=modulus_values,density_kg_m3=density,drag_width_m=width,
        peak_wind_m_s=1.,damping_ratio=damping,model="MuJoCo reduced beam coordinates",
        longest_beam_peak_mm=amplitude.max(1).tolist(),longest_beam_peak_s=t[amplitude.argmax(1)].tolist(),
        modal_validation=checks,longest_beam_index=tallest)
    out.joinpath("result.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    np.savez_compressed(out/"curves.npz",time_s=t,wind_m_s=wind(t),tip_m=states)

    fig=plt.figure(figsize=(10,4),layout="constrained")
    grid=fig.add_gridspec(2,3,width_ratios=[1,1,1.55],height_ratios=[1,1.55])
    plants=[fig.add_subplot(grid[:,i]) for i in range(2)]
    forcing=fig.add_subplot(grid[0,2])
    response=fig.add_subplot(grid[1,2],sharex=forcing)
    fig.suptitle("Lavender  |  Same gust, different stem stiffness",fontsize=16,weight="semibold")
    right=np.array([.976,.217]);right/=np.linalg.norm(right)
    surfaces=[]
    labels=["90 MPa · reference setting","360 MPa · stiff control"]
    palette=["#634392","#bd7360"]
    for j,ax in enumerate(plants):
        ax.set_title(f"{chr(97+j)}   {labels[j]}",loc="left",fontsize=9,color=palette[j])
        ax.set(xlim=(-.19,.22),ylim=(-.026,.48),aspect="equal")
        ax.axis("off")
        surface=PolyCollection([],linewidth=0,antialiased=False,rasterized=True)
        ax.add_collection(surface)
        surfaces.append(surface)
    forcing.plot(t,wind(t),color="#648681",lw=1.6)
    forcing.set(title="c   Prescribed gust and tip response",ylabel="Wind (m/s)")
    forcing.tick_params(labelbottom=False)
    for curve,color,label in zip(amplitude,palette,labels):
        response.plot(t,curve,color=color,lw=1.8,label=label.split(" · ")[0])
    response.set(xlabel="Time (s)",ylabel="Longest stem tip (mm)")
    response.legend(frameon=False,fontsize=7)
    for ax in (forcing,response):
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",color=".9",lw=.6)
    cursors=[ax.axvline(0,lw=.8,color=".6") for ax in (forcing,response)]
    caption=fig.supxlabel("",fontsize=8)
    plt.rcParams.update({"animation.ffmpeg_path":imageio_ffmpeg.get_ffmpeg_exe(),"pdf.fonttype":42})
    with FFMpegWriter(fps=24,codec="libx264",extra_args=["-pix_fmt","yuv420p"]).saving(
            fig,str(out/"lavender_wind.mp4"),dpi=144) as writer:
        for i in range(len(t)):
            for surface,state in zip(surfaces,states):
                tris, depths, colors = [], [], []
                for (xyz,faces,rgb,_,mode),displacements in zip(parts,state):
                    moving=xyz.copy()
                    moving[:,:2]+=right*mode[:,None]*displacements[i]
                    tri=moving[faces]
                    tris.append(np.stack([tri[:,:,:2]@right,tri[:,:,2]],axis=-1))
                    depths.append((tri[:,:,:2]@np.array([-right[1],right[0]])).mean(1))
                    colors.append(rgb)
                order=np.argsort(np.concatenate(depths))
                surface.set_verts(np.concatenate(tris)[order])
                surface.set_facecolor(np.concatenate(colors)[order])
            for cursor in cursors:
                cursor.set_xdata([t[i],t[i]])
            caption.set_text(f"{t[i]:.2f} s   ·   Fixed roots   ·   1× displacement   ·   MuJoCo beam coordinates")
            writer.grab_frame()
            if i==int(np.argmax(amplitude[0])):
                fig.savefig(out/"lavender_wind.png",dpi=250)
                fig.savefig(out/"lavender_wind.pdf",dpi=250)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("asset",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    run(a.asset,a.output)
