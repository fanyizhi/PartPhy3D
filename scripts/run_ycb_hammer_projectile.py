"""Compare mass distribution and a head material edit on the scanned YCB hammer."""
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
import trimesh


def moments(x, mass, pitch):
    centre=np.average(x,axis=0,weights=mass)
    relative=x-centre
    inertia=np.eye(3)*(np.dot(mass,(relative**2).sum(1))+mass.sum()*pitch**2/6)
    return centre,inertia-(relative*mass[:,None]).T@relative


def run(source,out,full_flight=False):
    out.mkdir(parents=True,exist_ok=True)
    mesh=trimesh.load_mesh(source,process=True)
    if not mesh.is_volume:
        raise ValueError("Expected closed original YCB scan")
    raw=mesh.vertices-mesh.center_mass
    _,_,axes=np.linalg.svd(raw,full_matrices=False)
    vertices=raw@axes[[2,1,0]].T
    lower=vertices[:,2]<np.quantile(vertices[:,2],.10)
    upper=vertices[:,2]>np.quantile(vertices[:,2],.90)
    if np.ptp(vertices[lower,1])>np.ptp(vertices[upper,1]):
        vertices[:,2]*=-1
    vertices[:,2]-=vertices[:,2].min()
    oriented=trimesh.Trimesh(vertices,mesh.faces,process=False)
    trimesh.repair.fix_normals(oriented)
    pitch=.002
    x=oriented.voxelized(pitch).fill().points
    cut=.80*np.ptp(vertices[:,2])
    head=x[:,2]>=cut
    if min(head.sum(),(~head).sum())<50:
        raise ValueError("Head/handle partition is degenerate")
    dv=mesh.volume/len(x)
    face_region=vertices[mesh.faces,2].mean(1)>=cut
    head_point=x[head].mean(0)
    assigned=dv*np.where(head,7850.,1050.)
    uniform=np.full(len(x),assigned.sum()/len(x))
    edited=dv*np.where(head,2700.,1050.)
    np.testing.assert_array_equal(edited[~head],assigned[~head])
    impulse=assigned.sum()*np.array([0,1.25,2.45])
    angular=np.array([.039,0.,0.])
    t=np.linspace(0,1.40,1401) if full_flight else np.linspace(0,.48,481)
    gravity=np.array([0,0,-9.81])
    init=np.array([0,0,.50])
    rows,tracks=[],[]
    for label, mass in zip(["Uniform density","Steel head","Aluminum head edit"],[uniform,assigned,edited]):
        total=mass.sum()
        com,inertia=moments(x,mass,pitch)
        full=[inertia[i,j] for i,j in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))]
        string=lambda a:" ".join(map(str,a))
        model=mujoco.MjModel.from_xml_string(f"""<mujoco><option integrator="RK4" timestep=".001"/>
          <worldbody><body><freejoint/><inertial pos="0 0 0" mass="{total}"
          fullinertia="{string(full)}"/><site name="head" pos="{string(head_point-com)}" size=".004"/>
          </body></worldbody></mujoco>""")
        data=mujoco.MjData(model)
        data.qpos[:3]=init+com
        initial_v=impulse/total
        data.qvel[:3]=initial_v
        data.qvel[3:]=np.linalg.solve(inertia,angular)
        translations,rotations,marks=[],[],[]
        for i in range(len(t)):
            mujoco.mj_forward(model,data)
            rotation=np.empty(9)
            mujoco.mju_quat2Mat(rotation,data.qpos[3:7])
            translations.append(data.qpos[:3].copy())
            rotations.append(rotation.reshape(3,3))
            marks.append(data.site_xpos[0].copy())
            if i+1<len(t):
                mujoco.mj_step(model,data)
        translations=np.asarray(translations)
        ballistic=init+com+t[:,None]*initial_v+.5*t[:,None]**2*gravity
        error=float(np.max(np.abs(translations-ballistic)))
        if error>1e-9:
            raise ValueError(f"MuJoCo COM differs from gravity-only reference: {error}")
        rows.append(dict(method=label,mass_kg=float(total),com_m=com.tolist(),
                         inertia_kg_m2=inertia.tolist(),ballistic_error_m=error,
                         range_m=float(translations[-1,1]-translations[0,1])))
        tracks.append((com,translations,np.asarray(rotations),np.asarray(marks)))
    summary=dict(asset="YCB 048_hammer Google 16k",source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 partition="long-axis cut at 80 percent from handle end",head_cut_m=cut,voxel_pitch_m=pitch,
                 raw_voxel_volume_m3=float(len(x)*pitch**3),mesh_volume_m3=float(mesh.volume),
                 head_volume_m3=float(head.sum()*dv),handle_volume_m3=float((~head).sum()*dv),
                 impulse_ns=impulse.tolist(),angular_impulse_kg_m2_s=angular.tolist(),models=rows,
                 assigned_uniform_head_track_separation_mm=float(np.linalg.norm(tracks[1][-1]-tracks[0][-1],axis=1).max()*1000))
    (out/"result.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    np.savez_compressed(out/"tracks.npz",time_s=t,com_m=[item[1] for item in tracks],
                        head_m=[item[-1] for item in tracks],rotation=[item[2] for item in tracks])
    np.savez_compressed(out/"render_geometry.npz",vertices=vertices,faces=mesh.faces,
                        head=face_region,centres=[item[0] for item in tracks],head_point=head_point)
    if full_flight:
        return
    plt.rcParams.update({"animation.ffmpeg_path":imageio_ffmpeg.get_ffmpeg_exe(),"pdf.fonttype":42,"font.size":9})
    fig=plt.figure(figsize=(11,5),layout="constrained")
    grid=fig.add_gridspec(2,3,height_ratios=[1.35,1])
    axes=[fig.add_subplot(grid[0,i]) for i in range(3)]
    path_ax=fig.add_subplot(grid[1,0])
    track_ax=fig.add_subplot(grid[1,1:])
    fig.suptitle("YCB scanned hammer  |  Mass distribution and a local head edit",fontsize=15,weight="semibold")
    palette=["#747d85","#207777","#b0634c"]
    patches=[]
    dots=[]
    for i,((com,trajectory,_,mark),ax,color) in enumerate(zip(tracks,axes,palette)):
        ax.set(title=rows[i]["method"],xlim=(-.30,.30),ylim=(-.27,.24),aspect="equal")
        ax.axis("off")
        item=PolyCollection([],linewidths=0,antialiased=False,rasterized=True)
        ax.add_collection(item)
        a,=ax.plot([],[],"o",c=color,mec="white",ms=6,zorder=5)
        b,=ax.plot([],[],"x",c="#1e2935",ms=6,zorder=5)
        dots.append((a,b))
        patches.append(item)
        relative=trajectory-trajectory[0]
        path_ax.plot(*relative[:,1:].T,color=color,lw=1.9,ls=["--","-","-"][i],label=rows[i]["method"])
        delta=mark-mark[0]
        track_ax.plot(*delta[:,1:].T,color=color,lw=1.9,label=rows[i]["method"])
    for ax,title in [(path_ax,"COM displacement"),(track_ax,"Same head marker, relative to release")]:
        ax.set(title=title,xlabel="Downrange (m)",ylabel="Vertical motion (m)")
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False,fontsize=7)
    footer=fig.supxlabel("",fontsize=8)
    camera=np.array([[.36,.08,.93],[.93,-.08,-.36],[0,.994,-.10]])
    with FFMpegWriter(fps=24,codec="libx264",extra_args=["-pix_fmt","yuv420p"]).saving(
            fig,str(out/"ycb_hammer.mp4"),dpi=144) as writer:
        for i in range(0,len(t),10):
            for j,(centre,position,rotation,mark) in enumerate(tracks):
                # A side view retains world downrange and height for overlaid paths.
                world=(vertices-centre)@rotation[i].T+position[i]
                projection=np.column_stack((world[:,1]-position[i,1]+.20*(world[:,0]-position[i,0]),
                                            world[:,2]-position[i,2]+.15*(world[:,0]-position[i,0])))
                order=np.argsort(-world[mesh.faces,0].mean(1))
                patches[j].set_verts(projection[mesh.faces][order])
                colors=np.where(face_region[:,None],np.array([.29,.39,.48]) if j!=2 else
                                np.array([.69,.76,.80]),np.array([.67,.45,.27]))
                normals=oriented.face_normals@rotation[i].T
                brightness=.72+.28*np.maximum(0,normals@camera[0])
                patches[j].set_facecolor((colors*brightness[:,None])[order])
                dots[j][0].set_data([0],[0])
                offset=mark[i]-position[i]
                dots[j][1].set_data([offset[1]+.20*offset[0]],[offset[2]+.15*offset[0]])
            footer.set_text(f"{t[i]:.2f} s   ·   Upper views follow COM   ·   Dot: estimated COM   ·   Cross: fixed head marker   ·   Matched launch impulses")
            writer.grab_frame()
            if i==340:
                for fmt in ("png","pdf"):fig.savefig(out/f"ycb_hammer.{fmt}",dpi=280)
    print(summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--full-flight",action="store_true")
    a=p.parse_args()
    run(a.source,a.out,a.full_flight)

