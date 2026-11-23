"""Export and record a native MuJoCo two-body comparison."""
import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np


def record(result_file, out):
    result = json.loads(result_file.read_bytes())
    if result["scene"] != "designed_rubber_mallet":
        raise ValueError("Expected the designed mallet mass distributions")
    out.mkdir(parents=True,exist_ok=True)
    pivot = np.array(result["pivot_m"])
    def fmt(values):
        return " ".join(map(str,values))
    bodies=[]
    for i,(model,side) in enumerate(zip(result["models"],[-.18,.18])):
        tensor=np.array(model["inertia_kg_m2"])
        inertia=[tensor[a,b] for a,b in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]]
        centre=np.array(model["com_m"])-pivot
        parts=[]
        for size,pos,density,color in result["parts"]:
            parts.append(f'<geom type="box" size="{fmt(np.array(size)/2)}" pos="{fmt(np.array(pos)-pivot)}" rgba="{fmt(np.array(color)/255)}" contype="0" conaffinity="0"/>')
        marker=[".01 .72 .68 1","1 .36 .08 1"][i]
        bodies.append(f'<body name="{["assigned_materials","uniform_density"][i]}" pos="{side} 0 .5">'
            f'<joint type="hinge" axis="1 0 0" ref="0"/>'
            f'<inertial pos="{fmt(centre)}" mass="{result["mass_kg"]}" fullinertia="{fmt(inertia)}"/>'
            f'<site name="com_{i}" pos="{fmt(centre)}" rgba="{marker}" size=".012"/>'
            + "".join(parts) + "</body>")
    release=np.deg2rad(result["release_deg"])
    xml=f"""<mujoco model="PartPhy3D mallet COM">
  <compiler angle="radian"/>
  <option timestep=".001" integrator="RK4"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="2048"/></visual>
  <asset><texture name="ground" type="2d" builtin="checker" rgb1=".82 .86 .88" rgb2=".89 .92 .94" width="128" height="128"/>
    <material name="ground" texture="ground" texrepeat="7 7" reflectance=".08"/>
  </asset>
  <worldbody>
    <light pos="0 -1.5 2" dir="0 .6 -1" diffuse=".86 .86 .86"/>
    <geom type="plane" size="1 1 .01" material="ground" contype="0" conaffinity="0"/>
    <geom type="capsule" fromto="-.28 0 .5 .28 0 .5" size=".006" rgba=".19 .28 .32 1" contype="0" conaffinity="0"/>
    {''.join(bodies)}
  </worldbody>
  <keyframe><key name="release_28deg" qpos="{release} {release}"/></keyframe>
</mujoco>"""
    (out/"mallet_com.xml").write_text(xml,encoding="utf-8")
    model=mujoco.MjModel.from_xml_string(xml)
    data=mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,0)
    mujoco.mj_forward(model,data)
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[0,0,.30]
    camera.distance=1.19
    camera.azimuth=112
    camera.elevation=-13
    option=mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_JOINT]=False
    source=np.load(result_file.with_name("tracks.npz"),allow_pickle=False)["angle_rad"]
    with mujoco.Renderer(model,width=1280,height=720) as renderer:
        writer=imageio_ffmpeg.write_frames(str(out/"mallet_native_mujoco.mp4"),(1280,720),fps=30,
            codec="libx264",pix_fmt_out="yuv420p",ffmpeg_log_level="error")
        writer.send(None)
        try:
            for step in range(1601):
                np.testing.assert_allclose(data.qpos,source[:,step],atol=1e-9,rtol=0)
                if step%25==0:
                    renderer.update_scene(data,camera,scene_option=option)
                    writer.send(renderer.render())
                if step<1600:
                    mujoco.mj_step(model,data)
        finally:
            writer.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("result",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    record(a.result,a.output)
