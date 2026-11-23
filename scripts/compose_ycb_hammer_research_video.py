"""Combine Blender scan close-ups and MuJoCo world-space trajectories."""
import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
from PIL import Image


def run(frames_dir,data_dir,out):
    frames=sorted(frames_dir.glob("closeup_*.png"))
    with np.load(data_dir/"tracks.npz",allow_pickle=False) as archive:
        time,com,head=[archive[k] for k in ("time_s","com_m","head_m")]
    if len(frames)!=len(time[::10]):
        raise ValueError("Blender frames and MuJoCo timestamps do not match")
    summary=json.loads((data_dir/"result.json").read_bytes())
    out.mkdir(parents=True,exist_ok=True)
    colors=["#65717c","#158479","#b7634a"]
    labels=["Uniform density","Steel head","Aluminum head edit"]
    plt.rcParams.update({"pdf.fonttype":42,"font.size":9,"animation.ffmpeg_path":imageio_ffmpeg.get_ffmpeg_exe()})
    fig=plt.figure(figsize=(12,6.75),facecolor="#f6f9fa")
    fig.subplots_adjust(left=.075,right=.96,top=.84,bottom=.13,wspace=.26,hspace=.24)
    grid=fig.add_gridspec(2,2,height_ratios=[1.34,1])
    fig.text(.075,.923,"Local material editing changes rigid-body motion",fontsize=21,
             weight="semibold",color="#162c35")
    fig.text(.076,.879,"YCB 048  /  Scanned geometry  /  MuJoCo flight with matched impulses",
             fontsize=10,color="#536c76")
    render=fig.add_subplot(grid[0,:])
    render.set_position([.075,.445,.885,.344])
    render.axis("off")
    def read_frame(path):
        with Image.open(path) as frame:
            # Preserve aspect and crop only the vertical margins around the aligned meshes.
            return np.asarray(frame.convert("RGBA").crop((0,4,1200,466)))
    pose=render.imshow(read_frame(frames[0]),aspect="equal")
    for j,(label,color,item) in enumerate(zip(labels,colors,summary["models"])):
        x=[.358,.518,.677][j]
        fig.text(x,.825,label,ha="center",fontsize=11,weight="semibold",color=color)
        fig.text(x,.796,f'{item["mass_kg"]:.3f} kg',ha="center",fontsize=8,color="#536c76")

    curves=[]
    for col,(track,title) in enumerate([(com,"a   Centre-of-mass displacement"),(head,"b   Identical head marker")]):
        ax=fig.add_subplot(grid[1,col])
        ax.set_title(title,loc="left",weight="semibold")
        ax.set(xlabel="Downrange (m)",ylabel="Vertical displacement (m)")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(color="#e0e9eb",lw=.55)
        marks=[]
        for j,(label,color) in enumerate(zip(labels,colors)):
            values=track[j,:,1:]-track[j,0,1:]
            style="--" if (col,j)==(0,0) else "-"
            ax.plot(*values.T,style,color=color,lw=1.9,label=label,
                    zorder=4 if (col,j)==(0,0) else 3)
            mark,=ax.plot([],[],"o",color=color,mec="white",ms=5,zorder=6)
            marks.append((mark,values))
        if col==0:
            ax.legend(loc="upper left",fontsize=7,frameon=False)
        curves.append(marks)
    footer=fig.text(.075,.040,"",fontsize=9,color="#425e68")
    with FFMpegWriter(fps=24,codec="libx264",extra_args=["-pix_fmt","yuv420p","-crf","17"]).saving(
            fig,str(out/"ycb_hammer_research.mp4"),dpi=144) as writer:
        for i,file in enumerate(frames):
            k=i*10
            pose.set_data(read_frame(file))
            for marks in curves:
                for mark,values in marks:
                    mark.set_data([values[k,0]],[values[k,1]])
            footer.set_text(f"{time[k]:0.2f} s   ·   COM-follow close-ups   ·   Colored dot: estimated COM   ·   Gold dot: head marker")
            repeats=20 if i in (0,len(frames)-1) else 2
            for _ in range(repeats):
                writer.grab_frame()
            if k==340:
                for ext in ("png","pdf"):
                    fig.savefig(out/f"ycb_hammer_research.{ext}",dpi=250)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("frames","data","output"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    run(args.frames,args.data,args.output)

