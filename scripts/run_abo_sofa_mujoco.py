"""Compare mass-matched rigid dynamics of an ABO sofa in MuJoCo."""
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
import mujoco
import numpy as np
from scipy.integrate import solve_ivp
import trimesh


def distribution(x, mass):
    centre = np.average(x, axis=0, weights=mass)
    r = x-centre
    inertia = np.eye(3)*np.dot(mass,np.sum(r*r,axis=1)) - (r*mass[:,None]).T@r
    return centre,inertia


def run(source, out, mallet=False):
    out.mkdir(parents=True,exist_ok=True)
    if mallet:
        # The handle ends at the base of the head so the material regions do not overlap.
        parts = [([.028,.032,.30],[0,0,.15],650.,[173,117,68,255]),
                 ([.16,.074,.082],[0,0,.341],1100.,[57,75,84,255])]
        meshes, positions, volumes, masses = [], [], [], []
        for size, centre, density, color in parts:
            body = trimesh.creation.box(size)
            body.apply_translation(centre)
            body.visual.vertex_colors = color
            meshes.append(body)
            count = np.ceil(np.array(size)/.004).astype(int)
            nodes = np.meshgrid(*[(np.arange(n)+.5)*d/n-d/2+c
                                  for n,d,c in zip(count,size,centre)],indexing="ij")
            pos = np.stack(nodes,axis=-1).reshape(-1,3)
            vol = np.full(len(pos),float(np.prod(size))/len(pos))
            positions.append(pos)
            volumes.append(vol)
            masses.append(vol*density)
        mesh=trimesh.util.concatenate(meshes)
        x,volume,mass = [np.concatenate(items) for items in (positions,volumes,masses)]
        total = mass.sum()
        true_com=np.average([p[1] for p in parts],axis=0,
                            weights=[np.prod(p[0])*p[2] for p in parts])
        np.testing.assert_allclose(np.average(x,axis=0,weights=mass),true_com,atol=1e-12)
        provenance=dict(geometry="two disjoint cuboids",parts=parts,analytic_com_m=true_com.tolist())
        title="Mallet  |  Material assignment and rigid-body motion"
    else:
        particle_file = source/"sofa_mpm_input.npz"
        mesh_file = source/"corrected_camera_space.ply"
        with np.load(particle_file,allow_pickle=False) as stored:
            x,mass,volume = [stored[k] for k in ("position_m","mass_kg","volume_m3")]
        mesh=trimesh.load_mesh(mesh_file,process=False)
        total = mass.sum()
        provenance=dict(mesh_sha256=hashlib.sha256(mesh_file.read_bytes()).hexdigest(),
                        particle_sha256=hashlib.sha256(particle_file.read_bytes()).hexdigest())
        title="Same sofa, same mass, different mass distribution"
    cases = [distribution(x,mass),distribution(x,total*volume/volume.sum())]
    pivot = cases[1][0]+[0,0,.115 if mallet else .21]
    dt, release, duration = .001, np.deg2rad(28), 1.6
    times = np.linspace(0,duration,1601)
    paths, checks = [], []
    for centre,inertia in cases:
        relative = centre-pivot
        full = [inertia[i,j] for i,j in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]]
        vector = lambda a: " ".join(str(v) for v in a)
        model = mujoco.MjModel.from_xml_string(f"""<mujoco>
          <option timestep="{dt}" integrator="RK4" gravity="0 0 -9.81"/>
          <worldbody><body><joint type="hinge" axis="1 0 0"/>
          <inertial pos="{vector(relative)}" mass="{total}" fullinertia="{vector(full)}"/>
          </body></worldbody></mujoco>""")
        data = mujoco.MjData(model)
        data.qpos[0]=release
        angles=[release]
        for _ in times[1:]:
            mujoco.mj_step(model,data)
            angles.append(float(data.qpos[0]))
        pivot_I=inertia[0,0]+total*np.dot(relative[1:],relative[1:])
        def rhs(_, y):
            torque = -9.81*total*(relative[1]*np.cos(y[0])-relative[2]*np.sin(y[0]))
            return [y[1],torque/pivot_I]
        ref = solve_ivp(rhs,(0,duration),[release,0],t_eval=times,rtol=1e-10,atol=1e-12).y[0]
        error = float(np.max(abs(ref-angles)))
        if error>1e-6:
            raise ValueError(f"MuJoCo and independent rigid-body ODE disagree: {error}")
        paths.append(np.asarray(angles))
        checks.append(dict(com_m=centre.tolist(),inertia_kg_m2=inertia.tolist(),pivot_I_kg_m2=pivot_I,
                           max_angle_residual_rad=error))
    summary=dict(scene="designed_rubber_mallet" if mallet else "B07QFP4PV5_ATVPDKIKX0DER",mass_kg=total,pivot_m=pivot.tolist(),release_deg=28,
        models=checks,com_separation_mm=float(np.linalg.norm(cases[0][0]-cases[1][0])*1000),
        peak_angle_difference_deg=float(np.rad2deg(np.max(abs(paths[0]-paths[1])))),
        **provenance)
    (out/"result.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    def rotate(v,a):
        c,s=np.cos(a),np.sin(a)
        return v@np.array([[1.,0.,0.],[0.,c,s],[0.,-s,c]])+pivot
    com_tracks=np.array([[rotate(c-pivot,a) for a in p] for (c,_),p in zip(cases,paths)])
    np.savez_compressed(out/"tracks.npz",time_s=times,angle_rad=paths,com_m=com_tracks)
    local=mesh.vertices-pivot
    faces=mesh.faces
    colors=mesh.visual.vertex_colors[faces,:3].mean(1)/255
    right=np.array([.43,.903,0] if mallet else [.62,.784,0]); right/=np.linalg.norm(right)
    up=np.array([-right[1]*.27,right[0]*.27,np.sqrt(1-.27**2)])
    cam=np.stack([right,up,np.cross(right,up)],axis=1)
    plt.rcParams.update({"animation.ffmpeg_path":imageio_ffmpeg.get_ffmpeg_exe(),"pdf.fonttype":42,
                        "svg.fonttype":"none","font.size":9})
    fig=plt.figure(figsize=(10,4.4),layout="constrained")
    g=fig.add_gridspec(2,3,width_ratios=[1,1,1.15])
    axes=[fig.add_subplot(g[:,j]) for j in range(2)]
    com_ax=fig.add_subplot(g[0,2])
    angle=fig.add_subplot(g[1,2])
    fig.suptitle(title,fontsize=16,weight="semibold")
    palette=["#237b80","#a76544"]
    meshes,markers=[],[]
    projected_pivot=pivot@cam
    for j,(ax,label,color) in enumerate(zip(axes,["a   Assigned materials","b   Uniform density"],palette)):
        ax.set_title(label,loc="left")
        ax.set(xlim=(-.26,.26) if mallet else (-.94,.94),
               ylim=(-.02,.44) if mallet else (-.2,.87),aspect="equal")
        ax.axis("off")
        item=PolyCollection([],linewidths=0,antialiased=False,rasterized=True)
        ax.add_collection(item)
        meshes.append(item)
        ax.scatter(*projected_pivot[:2],c="#182d36",marker="+",s=56,zorder=4)
        marker,=ax.plot([],[],"o",color=color,mec="white",ms=7,zorder=5)
        markers.append(marker)
        trail=com_tracks[j]@cam
        ax.plot(*trail[:,:2].T,c=color,lw=1.6,zorder=3)
        rel=(com_tracks[j]-pivot)*1000
        com_ax.plot(rel[:,1],rel[:,2],c=color,lw=1.6,label=label[4:])
        angle.plot(times,np.rad2deg(paths[j]),c=color,lw=1.6)
    com_ax.set(xlabel="Lateral COM (mm)",ylabel="Vertical COM (mm)")
    com_ax.set_title("c   Centre-of-mass path",loc="left")
    com_ax.legend(loc="best",frameon=False,fontsize=7)
    angle.set(xlabel="Time (s)",ylabel="Hinge angle (°)")
    angle.set_title("d   Rotation",loc="left")
    for ax in (com_ax,angle):ax.spines[["top","right"]].set_visible(False)
    cursor=angle.axvline(0,color=".65",lw=.8)
    note=fig.supxlabel("＋ Fixed hinge     ● Centre of mass     ·     Gravity, 28° release",fontsize=9)
    with FFMpegWriter(fps=24,codec="libx264",extra_args=["-pix_fmt","yuv420p"]).saving(fig,str(out/"sofa_com.mp4"),dpi=140) as writer:
        peak=int(np.argmax(np.abs(paths[0]-paths[1])))
        for k in sorted(set(range(0,len(times),25))|{peak}):
            for j in range(2):
                tri=(rotate(local,paths[j][k])@cam)[faces]
                order=np.argsort(tri[:,:,2].mean(1))
                meshes[j].set_verts(tri[order,:,:2])
                meshes[j].set_facecolor(colors[order])
                a,b=(com_tracks[j,k]@cam)[:2]
                markers[j].set_data([a],[b])
            cursor.set_xdata([times[k],times[k]])
            note.set_text(f"{times[k]:.2f} s   ·   + Fixed hinge     ● Centre of mass   ·   Gravity-only release")
            writer.grab_frame()
            if k==peak:
                for ext in ("png","pdf"):fig.savefig(out/f"sofa_com.{ext}",dpi=280)
    plt.close(fig)
    print(summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--mallet",action="store_true")
    a=p.parse_args()
    run(a.source,a.output,a.mallet)

