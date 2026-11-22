"""Reconstruct spatial material points and simulate an ABO cushion impact."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
import trimesh

from tactile3d.physics.materials import load_material_priors
from tactile3d.physics.mesh_particles import Construction, sample_mesh_particles
from tactile3d.physics.mpm_validation import coarsen_material_points, run_mpm

SCENE = "B074VM1T62"


def run(mesh_path, proposal_path, rgb_path, output):
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads(proposal_path.read_bytes())
    mesh_digest = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    if mesh_digest != report["mesh_sha256_before_winding_repair"] or report["scene_id"] != SCENE:
        raise ValueError("Mesh or proposal does not match the recorded ABO object")
    mesh = trimesh.load_mesh(mesh_path, process=False)
    trimesh.repair.fix_normals(mesh, multibody=True)
    # Reconstruction y is the short axis. Rotate it upward for face-down loading.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1,0,0]))
    mesh.apply_scale(report["predicted_max_length_m"]/mesh.extents.max())
    shell = next(p for p in report["parts"] if p["name"] == "outer fabric shell")
    core = next(p for p in report["parts"] if p["name"] == "inner foam core")
    layout = Construction.from_visual_hypothesis(shell["material"], "foam_filled",
                  core=core["material"], thickness_m=shell["thickness_mm"]/1000)
    points = sample_mesh_particles(mesh, layout, load_material_priors("configs/material_priors.yaml"),
                                  finest_m=.0015, bulk_m=.02, poisson={"fabric":.33,"foam":.25})
    points = coarsen_material_points(points,.017)
    x = points["position_m"]
    frames = [x]
    def capture(step, _, pos):
        if step % 30 == 0:
            frames.append(pos)
    speed, dt, steps = -3., 8e-6, 1500
    floor = x[:,2].min()+1e-9
    result = run_mpm(points, dx=.034, steps=steps, timestep_s=dt, floor_z_m=floor,
                     initial_vertical_speed=speed, on_step=capture)
    positions = np.stack(frames)
    times = np.arange(len(frames))*30*dt*1000
    heights = np.ptp(positions[:,:,2],axis=1)*1000
    stats = dict(scene=SCENE, particles=len(x), total_mass_kg=float(points["mass_kg"].sum()),
                 shell_mm=shell["thickness_mm"], speed_m_s=speed, grid_m=.034, dt_s=dt,
                 initial_height_mm=float(heights[0]), min_height_mm=float(heights.min()),
                 final_height_mm=float(heights[-1]), time_of_min_ms=float(times[heights.argmin()]),
                 final_min_J=float(result["jacobian"].min()), mesh_sha256=mesh_digest,
                 rgb_sha256=hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
                 p2g_residual_kg=result["max_p2g_mass_residual_kg"])
    (output/"result.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    np.savez_compressed(output/"trajectory.npz",**points,time_ms=times,trajectory_m=positions)
    render(rgb_path,output,positions,points,times,heights,floor)
    print(stats,flush=True)


def render(rgb, output, frames, points, times, heights, floor):
    with Image.open(rgb) as im:
        im = im.convert("RGBA")
        im = im.crop(im.getchannel("A").getbbox())
        photo = Image.alpha_composite(Image.new("RGBA",im.size,"white"),im)
    plt.rcParams.update({"pdf.fonttype":42,"svg.fonttype":"none","font.size":8})
    fig,axes=plt.subplots(1,3,figsize=(8.7,2.7),layout="constrained")
    ax,cut,response=axes
    ax.imshow(photo)
    ax.axis("off")
    ax.set_title("a   ABO-500 input",loc="left",weight="semibold")
    x=frames[0]
    take=np.abs(x[:,1]-np.median(x[:,1])) < .017
    col=np.where(points["material_id"][take]==0,"#9b5871","#43857f")
    # Project a central material-point slice. The light outline is the projected hull of all particles.
    outline,=cut.plot([],[],color="#aeb8bb",lw=.8,zorder=1)
    dots=cut.scatter(x[take,0]*1000,x[take,2]*1000,s=7,c=col,lw=0,zorder=2)
    cut.axhline(floor*1000,lw=.8,color="#46565d")
    cut.set(xlim=(x[:,0].min()*1000-12,x[:,0].max()*1000+12),
            ylim=(floor*1000-20,x[:,2].max()*1000+24),aspect="equal")
    cut.axis("off")
    cut.set_title("b   Material points · side section",loc="left",weight="semibold")
    status=cut.text(.5,.01,"",transform=cut.transAxes,ha="center",fontsize=8)
    cut.text(.50,.92,"Fabric",color="#9b5871",transform=cut.transAxes,ha="right")
    cut.text(.54,.92,"Foam",color="#43857f",transform=cut.transAxes,ha="left")
    response.plot(times,heights,color="#176e68",lw=1.8)
    marker,=response.plot([],[],"o",color="#176e68",ms=4)
    response.set(xlabel="Time (ms)",ylabel="Particle envelope height (mm)")
    response.set_title("c   Compression and recovery",loc="left",weight="semibold")
    response.spines[["top","right"]].set_visible(False)
    images=[]
    for j,pos in enumerate(frames):
        p=pos[:,[0,2]]*1000
        hull=ConvexHull(p).vertices
        outline.set_data(*p[np.r_[hull,hull[0]]].T)
        dots.set_offsets(p[take])
        status.set_text(f"{times[j]:.2f} ms  ·  motion ×1")
        marker.set_data([times[j]],[heights[j]])
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
        if j==heights.argmin():
            for ext in ("png","pdf","svg"):
                fig.savefig(output/f"pillow.{ext}",dpi=300)
    images[0].save(output/"pillow.gif",save_all=True,append_images=images[1:],
                   duration=[70]*(len(images)-1)+[700],loop=0,disposal=2)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("mesh","proposal","rgb","output"):
        parser.add_argument(arg,type=Path)
    a=parser.parse_args()
    run(a.mesh,a.proposal,a.rgb,a.output)

