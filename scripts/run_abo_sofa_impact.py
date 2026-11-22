"""Run a paired MLS-MPM contact pilot on the reconstructed ABO-500 grey sofa."""
import argparse
import hashlib
import json
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

SCENE = "B07QFP4PV5_ATVPDKIKX0DER"
GRID = 0.12307692307692308
DT = 4.8e-6
SPEED = -2.
STEPS = 800


def simulate(points, floor):
    snapshots = [points["position_m"]]
    def capture(step, _, position):
        if step % 20 == 0:
            snapshots.append(position)
    result = run_mpm(points, dx=GRID, steps=STEPS, timestep_s=DT,
                     initial_vertical_speed=SPEED, floor_z_m=floor, on_step=capture)
    return np.stack(snapshots), result


def main(folder, rgb_path, out):
    if rgb_path.name != f"{SCENE}_00.png":
        raise ValueError("Expected the fixed (00) view of the ABO-500 sofa")
    out.mkdir(parents=True, exist_ok=True)
    paths = [rgb_path, folder/"corrected_camera_space.ply", folder/"sofa_mpm_input.npz"]
    with np.load(paths[2]) as file:
        points = dict(file)
    x, mass = points["position_m"], points["mass_kg"]
    materials = points["material_names"][points["material_id"]]
    np.testing.assert_allclose(mass, points["density_kg_m3"]*points["volume_m3"])
    foam_index = np.flatnonzero(materials == "foam")[0]
    control = points | {key: np.full(len(x), points[key][foam_index])
                        for key in ("youngs_pa", "poisson")}
    floor = x[:,2].min()+1e-9
    assigned, result = simulate(points, floor)
    uniform, baseline = simulate(control, floor)
    times = np.arange(len(assigned))*20*DT*1000
    rms = lambda tr: np.sqrt(np.average(np.sum((tr-x)**2, axis=2), axis=1, weights=mass))*1000
    # Remove common translation from the heterogeneous response for the field overlay.
    delta = assigned-x
    local = np.linalg.norm(delta-np.average(delta, axis=1, weights=mass)[:,None],axis=2)*1000
    np.savez_compressed(out/"tracks.npz", assigned_m=assigned, uniform_m=uniform, time_ms=times,
                        local_displacement_mm=local, assigned_rms_mm=rms(assigned), uniform_rms_mm=rms(uniform))
    def metrics(r):
        return dict(energy_j=r["elastic_energy_j"], min_J=float(r["jacobian"].min()),
                    p2g_residual_kg=r["max_p2g_mass_residual_kg"])
    protocol = dict(scene=SCENE, rgb_view=0, grid_m=GRID, dt_s=DT, steps=STEPS, speed_m_s=SPEED,
                    particles=len(x), mass_kg=float(mass.sum()), floor_z_m=float(floor),
                    assigned=metrics(result), uniform_foam=metrics(baseline),
                    max_local_mm=float(local.max()), input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out/"result.json").write_text(json.dumps(protocol, indent=2),encoding="utf-8")

    with Image.open(rgb_path) as im:
        rgba = im.convert("RGBA")
        rgba = rgba.crop(rgba.getchannel("A").getbbox())
        photo = Image.alpha_composite(Image.new("RGBA",rgba.size,"white"),rgba).convert("RGB")
    mesh = trimesh.load_mesh(paths[1], process=False)
    right = np.array([.94,.342,0])
    right /= np.linalg.norm(right)
    up = np.array([-right[1]*.30,right[0]*.30,np.sqrt(1-.30**2)])
    camera = np.stack([right,up,np.cross(right,up)],axis=1)
    projected = x@camera
    triangles = (mesh.vertices@camera)[mesh.faces]
    order = np.argsort(triangles[:,:,2].mean(1))
    surface = mesh.visual.vertex_colors[mesh.faces,:3].mean(1)/255
    plt.rcParams.update({"font.size":8,"pdf.fonttype":42,"svg.fonttype":"none"})
    fig, axs = plt.subplots(1,4,figsize=(10,2.75),gridspec_kw={"width_ratios":[1,1,1,1.2]})
    fig.subplots_adjust(left=.02,right=.984,bottom=.245,top=.80,wspace=.23)
    axs[0].imshow(photo)
    ax = axs[1]
    ax.add_collection(PolyCollection(triangles[order,:,:2],facecolors=surface[order],
                                     edgecolors="face",lw=.12,alpha=.15,rasterized=True))
    colors = {"fabric":"#446b9f","foam":"#409687","wood":"#bb743d"}
    chosen = np.flatnonzero(x[:,1]>=np.median(x[:,1]))
    chosen = chosen[np.argsort(projected[chosen,2])]
    ax.scatter(*projected[chosen,:2].T,s=5,c=[colors[m] for m in materials[chosen]],lw=0,rasterized=True)
    ax.legend(handles=[Line2D([],[],marker="o",lw=0,color=c,label=m.title())
                        for m,c in colors.items()],loc="upper center",bbox_to_anchor=(.5,-.03),
              frameon=False,ncol=3,fontsize=7,handletextpad=.15,columnspacing=.5)
    order = np.argsort(projected[:,2])
    dots = axs[2].scatter(*projected[order,:2].T,s=4,c=local[0,order],cmap="magma",
                          vmin=0,vmax=local.max(),lw=0,rasterized=True)
    for ax,title in zip(axs,["a  ABO-500 input","b  Mesh and interior","c  MLS-MPM impact","d  Paired response"]):
        ax.set_title(title,loc="left",weight="semibold",fontsize=9,pad=9)
    for ax in axs[:3]:
        ax.set_axis_off()
    for ax in axs[1:3]:
        ax.set(xlim=(projected[:,0].min()-.06,projected[:,0].max()+.06),
               ylim=(projected[:,1].min()-.07,projected[:,1].max()+.07),aspect="equal")
    cb=fig.colorbar(dots,cax=fig.add_axes([.52,.17,.19,.023]),orientation="horizontal")
    cb.set_label("Motion relative to COM (mm)")
    ax=axs[3]
    ax.plot(times,rms(assigned),color="#137773",label="Assigned")
    ax.plot(times,rms(uniform),"--",color="#975a47",label="Uniform foam")
    ax.set(xlabel="Time (ms)",ylabel="RMS displacement (mm)")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False,fontsize=7)
    cursor=ax.axvline(0,color=".72",lw=.7,zorder=0)
    status=fig.text(.025,.16,"",fontsize=8)
    frames=[]
    for j,time in enumerate(times):
        dots.set_offsets((assigned[j]@camera)[order,:2])
        dots.set_array(local[j,order])
        cursor.set_xdata([time,time])
        status.set_text(f"Vertical impact · {time:.2f} ms · geometry ×1")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
    for ext in ("png","pdf","svg"):
        fig.savefig(out/f"abo_sofa_impact.{ext}",dpi=330)
    frames[0].save(out/"abo_sofa_impact.gif",save_all=True,append_images=frames[1:],
                   duration=[75]*(len(frames)-1)+[850],loop=0,disposal=2)
    print(json.dumps(protocol,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("folder","rgb","out"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    main(args.folder,args.rgb,args.out)

