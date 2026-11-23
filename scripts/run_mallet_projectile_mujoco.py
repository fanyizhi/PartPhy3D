"""Render a gravity-only mallet projectile in MuJoCo with synchronized curves."""
import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import mujoco
import numpy as np


def run(source, out, edit=False):
    out.mkdir(parents=True,exist_ok=True)
    results=json.loads(source.read_bytes())
    dt=.001
    release_height=.78 if edit else .42
    velocity=np.array([0.,1.3,2.6])
    angular_momentum=.061
    offsets=[-.19,.18]
    original_parts=results["parts"]
    edited_parts=[[*part] for part in original_parts]
    edited_parts[1][2:]=[2700.,[156,178,195,255]]

    def properties(parts):
        size=np.array([part[0] for part in parts])
        pos=np.array([part[1] for part in parts])
        mass=np.prod(size,axis=1)*[part[2] for part in parts]
        centre=np.average(pos,axis=0,weights=mass)
        offset=pos-centre
        diagonal=((size**2).sum(1)[:,None]-size**2)/12
        diagonal+=(offset**2).sum(1)[:,None]-offset**2
        return float(mass.sum()),centre,mass@diagonal

    if edit:
        assert edited_parts[0]==original_parts[0]
        assert edited_parts[1][:2]==original_parts[1][:2]
        variants=[(parts,*properties(parts)) for parts in [original_parts,edited_parts]]
        labels=["Rubber head","Aluminum edit"]
    else:
        variants=[(original_parts,results["mass_kg"],np.array(r["com_m"]),
                   np.diag(r["inertia_kg_m2"])) for r in results["models"]]
        labels=["Assigned","Uniform density"]
    impulse=velocity*variants[0][1]
    velocities=np.array([impulse/mass for _,mass,_,_ in variants])
    bodies=[]
    def vector(a):
        return " ".join(map(str,a))
    for i,((parts,mass,centre,inertia),offset) in enumerate(zip(variants,offsets)):
        geoms="".join(f'<geom type="box" size="{vector(np.array(d)/2)}" pos="{vector(np.array(p)-centre)}" rgba="{vector(np.r_[np.array(c[:3])/255,.72])}" contype="0" conaffinity="0"/>'
                      for d,p,_,c in parts)
        marker=[".03 .82 .76 1","1 .39 .14 1"][i]
        bodies.append(f'<body name="mallet_{i}" pos="{offset} 0 {release_height}"><freejoint/>'
                      f'<inertial pos="0 0 0" mass="{mass}" diaginertia="{vector(inertia)}"/>'
                      f'<site name="com_{i}" pos="0 0 0" size=".015" rgba="{marker}"/>{geoms}</body>')
    duration=.45 if edit else .63
    time=np.arange(round(duration/dt)+1)*dt
    parabola=velocities[None,:,1:]*time[:,None,None]+np.array([0.,release_height])
    parabola[:,:,1]-=4.905*time[:,None]**2
    beads="".join('<site type="sphere" size=".004" rgba=".18 .32 .37 .4" pos="'+vector([x,*yz])+'"/>'
                  for j,x in enumerate(offsets) for yz in parabola[::25,j])
    velocity_key=[]
    for v,(_,_,_,inertia) in zip(velocities,variants):
        velocity_key.extend([*v,angular_momentum/inertia[0],0,0])
    xml=f"""<mujoco model="PartPhy3D projectile">
    <option timestep="{dt}" integrator="RK4" gravity="0 0 -9.81"/>
    <visual><global offwidth="1280" offheight="720"/><quality shadowsize="2048"/></visual>
    <asset><texture type="skybox" builtin="gradient" rgb1=".92 .95 .97" rgb2=".74 .81 .85" width="128" height="512"/><texture name="floor" type="2d" builtin="checker" rgb1=".81 .86 .89" rgb2=".88 .91 .93" width="128" height="128"/>
      <material name="floor" texture="floor" texrepeat="6 6"/></asset>
    <worldbody>
      <light pos="1 -.7 2" dir="-.5 .4 -1" diffuse=".9 .9 .9"/>
      <geom type="plane" size="2 2 .01" material="floor" contype="0" conaffinity="0"/>
      <camera name="overview" pos="1.85 -0.32 1.45" xyaxes="0.39 0.92 0 -.47 .20 .86"/>
      {beads}{''.join(bodies)}
    </worldbody>
    <keyframe><key name="launch" qvel="{vector(velocity_key)}"/></keyframe>
    </mujoco>"""
    (out/"mallet_projectile.xml").write_text(xml,encoding="utf-8")
    model=mujoco.MjModel.from_xml_string(xml)
    data=mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,0)
    mujoco.mj_forward(model,data)
    positions=[]
    rotations=[]
    states=[]
    for step in range(len(time)):
        positions.append(np.vstack([data.qpos[1:3],data.qpos[8:10]]))
        rotations.append([2*np.arctan2(data.qpos[4],data.qpos[3]),
                          2*np.arctan2(data.qpos[11],data.qpos[10])])
        if step%10==0:
            states.append(data.qpos.copy())
        if step+1<len(time):
            mujoco.mj_step(model,data)
    positions=np.asarray(positions)
    error=float(np.max(np.abs(positions-parabola)))
    if error>1e-9:
        raise ValueError(f"Ballistic trajectory mismatch: {error}")
    angles=np.rad2deg(np.unwrap(np.asarray(rotations),axis=0))
    np.savez_compressed(out/"tracks.npz",t_s=time,com_yz_m=positions,analytic_yz_m=parabola,angle_deg=angles)
    summary=dict(launch_velocity_m_s=velocities.tolist(),same_linear_impulse_ns=impulse.tolist(),
                 material_edit=edit, mass_kg=[v[1] for v in variants],
                 com_local_m=[v[2].tolist() for v in variants], inertia_x_kg_m2=[float(v[3][0]) for v in variants],
                 final_range_m=parabola[-1,:,0].tolist(),launch_height_m=release_height,
                 angular_momentum_kg_m2_s=angular_momentum,max_com_error_m=error,
                 final_angle_deg=angles[-1].tolist(),duration_s=duration)
    (out/"result.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    plt.rcParams.update({"animation.ffmpeg_path":imageio_ffmpeg.get_ffmpeg_exe(),"pdf.fonttype":42,"font.size":9})
    fig=plt.figure(figsize=(10,4.5),layout="constrained")
    grid=fig.add_gridspec(2,2,width_ratios=[1.7,1])
    scene=fig.add_subplot(grid[:,0]); pos=fig.add_subplot(grid[0,1]); angle=fig.add_subplot(grid[1,1])
    fig.suptitle("Local material edit  |  Matched-impulse projectile" if edit else "Mallet projectile  |  Centre-of-mass motion and rotation",fontsize=15,weight="semibold")
    frame=scene.imshow(np.zeros((720,1280,3),dtype=np.uint8))
    scene.axis("off")
    scene.set_title("a   Native MuJoCo render",loc="left")
    scene.text(.02,.04,f"Turquoise  {labels[0]}   ·   Orange  {labels[1]}",transform=scene.transAxes,color="#20343d",
               fontsize=9,bbox={"fc":"white","ec":"none","alpha":.85})
    palette=["#137d7f","#ad6544"]
    cursors=[]
    for j,(c,label) in enumerate(zip(palette,labels)):
        if edit or j==0:
            pos.plot(*parabola[:,j].T,color=c,lw=1.7,label=label if edit else "Analytic and both simulations")
        marker,=pos.plot([],[],"o",color=c,ms=4)
        cursors.append(marker)
        angle.plot(time,angles[:,j],color=c,lw=1.8,label=label)
    pos.set(xlabel="Downrange (m)",ylabel="COM height (m)",title="b   Centre-of-mass parabolas")
    pos.legend(frameon=False,fontsize=7)
    angle.set(xlabel="Time (s)",ylabel="Rotation (°)",title="c   Rotation at matched angular momentum")
    angle.legend(frameon=False,fontsize=7)
    cursor=angle.axvline(0,color=".55",lw=.7)
    caption=fig.supxlabel("",fontsize=9)
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[0,.25,.63] if edit else [0,.40,.47]
    camera.distance=1.95
    camera.azimuth=128
    camera.elevation=-16
    with mujoco.Renderer(model,width=1280,height=720) as renderer:
        with FFMpegWriter(fps=24,codec="libx264",extra_args=["-pix_fmt","yuv420p"]).saving(
                fig,str(out/"mallet_projectile.mp4"),dpi=144) as writer:
            for i,q in enumerate(states):
                k=i*10
                data.qpos[:]=q
                mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera)
                frame.set_data(renderer.render())
                for j,marker in enumerate(cursors):
                    marker.set_data([parabola[k,j,0]],[parabola[k,j,1]])
                cursor.set_xdata([time[k],time[k]])
                caption.set_text(f"{time[k]:.2f} s   ·   Gravity only   ·   "+
                    ("Matched linear and angular impulses" if edit else "Matched total mass and launch velocity"))
                writer.grab_frame()
                if k==(300 if edit else 380):
                    for ext in ("png","pdf"):fig.savefig(out/f"mallet_projectile.{ext}",dpi=280)
    print(summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--edit",action="store_true")
    a=p.parse_args()
    run(a.source,a.out,a.edit)
