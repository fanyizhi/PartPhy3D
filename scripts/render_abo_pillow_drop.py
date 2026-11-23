"""Compare assigned and reduced stiffness in a staged ABO cushion drop."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from PIL import Image

from tactile3d.physics.mpm_validation import run_mpm

SCENE = "B074VM1T62"
GRAVITY = 9.81
SCALE = .25
KEYS = ("position_m", "volume_m3", "density_kg_m3", "mass_kg", "youngs_pa",
        "poisson", "material_id", "material_names")


def run(source, image, output):
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((source/"result.json").read_bytes())
    if protocol["scene"] != SCENE or hashlib.sha256(image.read_bytes()).hexdigest() != protocol["rgb_sha256"]:
        raise ValueError("RGB does not match the recorded cushion")
    with np.load(source/"trajectory.npz", allow_pickle=False) as archive:
        points = {key: archive[key] for key in KEYS}
        reference = archive["trajectory_m"]
        time = archive["time_ms"]
    np.testing.assert_allclose(points["mass_kg"], points["density_kg_m3"]*points["volume_m3"])
    step_dt = protocol["dt_s"]
    stride = round((time[1]-time[0])*1e-3/step_dt)
    start = points["position_m"]
    floor = float(start[:,2].min())
    speed = abs(protocol["speed_m_s"])

    def simulate(stiffness_factor):
        frames = [start]
        def record(step, _, pos):
            if step % stride == 0:
                frames.append(pos)
        data = points | {"youngs_pa": points["youngs_pa"]*stiffness_factor}
        stats = run_mpm(data, dx=protocol["grid_m"], steps=(len(time)-1)*stride,
                        timestep_s=step_dt, floor_z_m=floor+1e-9,
                        initial_vertical_speed=-speed, on_step=record)
        return np.stack(frames), stats["max_p2g_mass_residual_kg"]

    assigned, residual_a = simulate(1.)
    np.testing.assert_allclose(assigned, reference, atol=1e-9, rtol=0)
    soft, residual_b = simulate(SCALE)
    contact = [assigned, soft]
    compression = [1000*(np.ptp(start[:,2])-np.ptp(x[:,:,2],axis=1)) for x in contact]
    free_end = speed/GRAVITY
    release_height = .5*GRAVITY*free_end**2
    run_info = dict(scene=SCENE, release_height_m=release_height, ballistic_duration_s=free_end,
                    contact_duration_ms=float(time[-1]), soft_E_factor=SCALE,
                    max_compression_mm=[float(d.max()) for d in compression],
                    particles=len(start), p2g_mass_residual_kg=[residual_a,residual_b],
                    source_trajectory_sha256=hashlib.sha256((source/"trajectory.npz").read_bytes()).hexdigest(),
                    materials={str(name): dict(n=int((points["material_id"]==i).sum()),
                        mass_kg=float(points["mass_kg"][points["material_id"]==i].sum()),
                        density_kg_m3=np.unique(points["density_kg_m3"][points["material_id"]==i]).tolist(),
                        E_pa=np.unique(points["youngs_pa"][points["material_id"]==i]).tolist(),
                        poisson=np.unique(points["poisson"][points["material_id"]==i]).tolist())
                        for i,name in enumerate(points["material_names"])})
    (output/"drop.json").write_text(json.dumps(run_info,indent=2),encoding="utf-8")
    np.savez_compressed(output/"drop_tracks.npz",contact_time_ms=time,assigned_m=assigned,
                        soft_m=soft,compression_mm=compression)
    with Image.open(image) as photo:
        photo = photo.convert("RGBA")
        rgb = np.asarray(photo.crop(photo.getchannel("A").getbbox()))
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42,"svg.fonttype":"none",
                        "animation.ffmpeg_path":imageio_ffmpeg.get_ffmpeg_exe()})
    fig, axes = plt.subplots(1,4,figsize=(12.8,4.8),
                            gridspec_kw={"width_ratios":[.85,1,1,1.4]},layout="constrained")
    fig.suptitle("ABO-500 cushion  |  Free fall and elastic response",fontsize=17,weight="semibold")
    for ax,title in zip(axes,["a  Dataset input","b  Assigned stiffness","c  Softer control (0.25× E)",
                              "d  Compression during contact"]):
        ax.set_title(title,loc="left",fontsize=11,weight="semibold")
    axes[0].imshow(rgb)
    axes[0].text(.5,-.08,SCENE+"  ·  view 00",ha="center",transform=axes[0].transAxes)
    axes[0].axis("off")
    mask = abs(start[:,1]-np.median(start[:,1]))<.017
    palette = {"fabric":"#bd6264","foam":"#338d87"}
    colors = [palette[str(n)] for n in points["material_names"][points["material_id"][mask]]]
    dots = []
    for ax in axes[1:3]:
        ax.set(xlim=(-280,280),aspect="equal")
        ax.axhline(0,color="#54666b",lw=1)
        dots.append(ax.scatter(start[mask,0]*1000,(start[mask,2]-floor)*1000,s=8,c=colors,lw=0))
        ax.set_xticks([])
        ax.set_ylabel("Height above floor (mm)")
        ax.spines[["top","right","bottom"]].set_visible(False)
        ax.text(.06,.95,"Fabric",color=palette["fabric"],transform=ax.transAxes,va="top")
        ax.text(.36,.95,"Foam",color=palette["foam"],transform=ax.transAxes,va="top")
    response=axes[3]
    for y,col,label in zip(compression,["#17756f","#a45256"],["Assigned","Softer control"]):
        response.plot(time,y,color=col,lw=2,label=label)
    response.axhline(0,color=".8",lw=.7)
    response.set(xlabel="Time since contact (ms)",ylabel="Envelope compression (mm)")
    response.legend(frameon=False)
    response.spines[["top","right"]].set_visible(False)
    cursor=response.axvline(0,color=".5",lw=.8)
    subtitle=fig.supxlabel("",fontsize=10)

    def update(positions, caption, y_max, time_cursor):
        for scatter,ax,xyz in zip(dots,axes[1:3],positions):
            scatter.set_offsets(np.column_stack([xyz[mask,0],xyz[mask,2]-floor])*1000)
            ax.set_ylim(-15,y_max)
        subtitle.set_text(caption)
        cursor.set_xdata([time_cursor,time_cursor])

    with FFMpegWriter(fps=24,codec="libx264",extra_args=["-pix_fmt","yuv420p","-crf","18"]).saving(
            fig,str(output/"pillow_drop.mp4"),dpi=150) as writer:
        for s in np.linspace(0,free_end,54):
            shifted=start+[0,0,release_height-.5*GRAVITY*s*s]
            update([shifted,shifted],f"Analytic free fall  ·  {1000*s:.0f} ms  ·  1× displacement",675,0)
            writer.grab_frame()
        for k,t in enumerate(time):
            update([assigned[k],soft[k]],f"MLS-MPM contact close-up  ·  {t:.2f} ms  ·  1× displacement",213,t)
            for _ in range(2):
                writer.grab_frame()
    k=int(np.argmax(compression[1]))
    update([assigned[k],soft[k]],f"MLS-MPM contact close-up  ·  {time[k]:.2f} ms  ·  1× displacement",213,time[k])
    for fmt in ("pdf","svg","png"):
        fig.savefig(output/f"pillow_drop.{fmt}",dpi=300)
    plt.close(fig)
    print(run_info,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("source","image","output"):
        parser.add_argument(arg,type=Path)
    args=parser.parse_args()
    run(args.source,args.image,args.output)
