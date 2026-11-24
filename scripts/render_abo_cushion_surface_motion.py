"""Show the reconstructed, colored ABO cushion driven by MLS-MPM particles."""
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
import numpy as np
from scipy.spatial import cKDTree
import trimesh


def render(mesh_file, particles_file, comparison, output, *, storyboard_only=False):
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((particles_file.parent/"result.json").read_bytes())
    drop = json.loads((comparison.parent/"drop.json").read_bytes())
    if hashlib.sha256(mesh_file.read_bytes()).hexdigest() != source["mesh_sha256"]:
        raise ValueError("Unexpected reconstruction mesh")
    if hashlib.sha256(particles_file.read_bytes()).hexdigest() != drop["source_trajectory_sha256"]:
        raise ValueError("Unexpected material point archive")
    with np.load(particles_file, allow_pickle=False) as stored:
        original = stored["position_m"]
    with np.load(comparison, allow_pickle=False) as stored:
        paths = [stored["assigned_m"], stored["soft_m"]]
        times = stored["contact_time_ms"]
        compression = stored["compression_mm"]
    for path in paths:
        np.testing.assert_array_equal(path[0], original)
    mesh = trimesh.load_mesh(mesh_file, process=False)
    if not mesh.visual.defined or mesh.visual.kind != "vertex":
        raise ValueError("Reconstruction must have original vertex colors")
    trimesh.repair.fix_normals(mesh, multibody=True)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
    mesh.apply_scale(.50/mesh.extents.max())
    vertices = np.asarray(mesh.vertices)
    faces = mesh.faces
    # Bind the visual surface to nearby textile particles, avoiding core averaging.
    with np.load(particles_file, allow_pickle=False) as stored:
        fabric_id, = np.flatnonzero(stored["material_names"]=="fabric")
        fabric = np.flatnonzero(stored["material_id"]==fabric_id)
    distance, neighbor = cKDTree(original[fabric]).query(vertices,k=8)
    weights = np.maximum(distance,1e-6)**-2
    weights /= weights.sum(1,keepdims=True)
    neighbor = fabric[neighbor]
    rest = vertices.copy()
    # The display floor is referenced to the particle floor, with the sub-particle gap visible.
    rest[:,2] -= original[:,2].min()
    rgb = mesh.visual.vertex_colors[faces,:3].mean(1)/255
    right=np.array([.958,.287,0])
    right/=np.linalg.norm(right)
    up=np.array([-right[1]*.30,right[0]*.30,np.sqrt(1-.30**2)])
    camera=np.stack([right,up,np.cross(right,up)],axis=1)
    ground=np.array([[-.58,-.38,0],[.58,-.38,0],[.58,.38,0],[-.58,.38,0]])@camera
    if storyboard_only:
        storyboard(output, rest, original, faces, rgb, camera, ground, neighbor,
                   weights, paths, times, compression, drop)
        return
    plt.rcParams.update({"animation.ffmpeg_path":imageio_ffmpeg.get_ffmpeg_exe()})
    fig, axes=plt.subplots(1,2,figsize=(10,5.25),facecolor="#f5f7f8")
    fig.subplots_adjust(left=.026,right=.974,top=.81,bottom=.14,wspace=.025)
    fig.text(.5,.945,"ABO-500  /  Cushion drop",ha="center",fontsize=21,weight="semibold",color="#152630")
    subtitle=fig.text(.5,.883,"",ha="center",fontsize=10,color="#475e6a")
    time_label=fig.text(.5,.075,"",ha="center",fontsize=11,color="#152630")
    fig.text(.5,.029,"B074VM1T62  ·  Same mesh, mass and release conditions  ·  Displacement ×1",
             ha="center",fontsize=9,color="#475e6a")
    surfaces=[]
    labels=[]
    for ax,title in zip(axes,["Assigned material properties","Softer control  ·  0.25× Young’s modulus"]):
        ax.set_facecolor("#e9eef0")
        ax.set_title(title,loc="center",fontsize=11,pad=10,color="#152630")
        ax.fill(*ground[:,:2].T,color="#d5dfe3",ec="#bacad1",lw=.8)
        surface=PolyCollection([],linewidth=0,antialiased=False,rasterized=True)
        ax.add_collection(surface)
        surfaces.append(surface)
        labels.append(ax.text(.04,.05,"",transform=ax.transAxes,fontsize=10,color="#152630"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        ax.set_aspect("equal")

    def update(deformed, limits):
        for ax,surface,xyz in zip(axes,surfaces,deformed):
            tri=(xyz@camera)[faces]
            normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            cosine=(normals@np.array([-.25,.65,.72]))/np.maximum(np.linalg.norm(normals,axis=1),1e-12)
            order=np.argsort(tri[:,:,2].mean(1))
            surface.set_verts(tri[order,:,:2])
            tint=.76+.24*np.clip(cosine,0,1)
            surface.set_facecolor((rgb*tint[:,None])[order])
            ax.set(xlim=limits[0],ylim=limits[1])

    with FFMpegWriter(fps=24,codec="libx264",extra_args=["-pix_fmt","yuv420p","-crf","17"]).saving(
            fig,str(output/"cushion_motion.mp4"),dpi=144) as writer:
        subtitle.set_text("Analytic free fall from 0.459 m  ·  Slow motion")
        duration=drop["ballistic_duration_s"]
        for t in np.linspace(0,duration,48):
            mesh_at_t=rest+[0,0,drop["release_height_m"]-4.905*t*t]
            update([mesh_at_t,mesh_at_t],[(-.43,.43),(-.14,.62)])
            time_label.set_text(f"{t*1000:05.1f} ms after release")
            writer.grab_frame()
        subtitle.set_text("MLS-MPM contact close-up  ·  Colored mesh follows surface particles")
        for i,t in enumerate(times):
            deformed=[rest+np.sum(weights[:,:,None]*(p[i]-original)[neighbor],axis=1) for p in paths]
            update(deformed,[(-.31,.31),(-.078,.258)])
            time_label.set_text(f"{t:05.2f} ms after contact")
            for label,curve in zip(labels,compression):
                label.set_text(f"Envelope compression   {curve[i]:+.2f} mm")
            writer.grab_frame()
            writer.grab_frame()
            if i == int(np.argmax(compression[1])):
                fig.savefig(output / "cushion_motion.png", dpi=170)
    plt.close(fig)


def storyboard(
    output,
    rest,
    reference,
    faces,
    face_rgb,
    camera,
    ground,
    neighbors,
    weights,
    paths,
    times,
    compression,
    drop,
):
    """Use identical floor coordinates in all four airborne/first-contact panels."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
        }
    )
    fig = plt.figure(figsize=(7.2, 4.7), facecolor="white")
    fig.text(
        0.06,
        0.965,
        "Striped cushion drop and contact response",
        va="top",
        weight="bold",
        size=13,
        color="#192b3b",
    )
    fig.text(
        0.06,
        0.894,
        "ABO-500  ·  Analytic free fall, then MLS-MPM at floor contact",
        size=7.5,
        color="#4f6574",
    )
    top = fig.add_gridspec(
        1, 4, left=0.06, right=0.967, top=0.824, bottom=0.48, wspace=0.11
    )

    def frame(ax, positions, limits):
        ax.set_facecolor("#eef3f5")
        ax.fill(*ground[:, :2].T, color="#d7e2e7", ec="#adbec9", lw=0.65)
        for x in np.linspace(-0.53, 0.53, 8):
            line = np.array([[x, -0.38, 0], [x, 0.38, 0]]) @ camera
            ax.plot(*line[:, :2].T, color="#bfcfd8", lw=0.35, zorder=2)
        tri = (positions @ camera)[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        cosine = (normals @ np.array([-0.25, 0.65, 0.72])) / np.maximum(
            np.linalg.norm(normals, axis=1), 1e-12
        )
        order = np.argsort(tri[:, :, 2].mean(1))
        tint = (0.76 + 0.24 * np.clip(cosine, 0, 1))[:, None]
        ax.add_collection(
            PolyCollection(
                tri[order, :, :2],
                facecolors=(face_rgb * tint)[order],
                linewidth=0,
                antialiased=False,
                rasterized=True,
                zorder=3,
            )
        )
        ax.set(xlim=limits[0], ylim=limits[1], xticks=[], yticks=[])
        ax.spines[:].set_visible(False)
        ax.set_aspect("equal", adjustable="box")

    end = drop["ballistic_duration_s"]
    for i, (fraction, title) in enumerate(
        zip(
            (0, 0.60, 0.84, 1.0),
            ("Release", "In flight", "Approach", "First contact"),
            strict=True,
        )
    ):
        ax = fig.add_subplot(top[i])
        t = end * fraction
        height = max(0, drop["release_height_m"] - 4.905 * t * t)
        frame(ax, rest + [0, 0, height], [(-0.36, 0.36), (-0.14, 0.67)])
        ax.set_title(f"{title}\n{t*1000:.0f} ms", fontsize=7.4, pad=4)
    fig.text(
        0.061,
        0.437,
        "Peak compression after contact",
        size=8.3,
        weight="bold",
        color="#192b3b",
    )
    lower = fig.add_gridspec(
        1,
        3,
        left=0.06,
        right=0.95,
        top=0.404,
        bottom=0.13,
        wspace=0.33,
        width_ratios=(1, 1, 1.37),
    )
    for i, (path, title) in enumerate(
        zip(paths, ("Assigned stiffness", "0.25× Young’s modulus"), strict=True)
    ):
        step = int(np.argmax(compression[i]))
        moved = rest + np.sum(
            weights[:, :, None] * (path[step] - reference)[neighbors], axis=1
        )
        ax = fig.add_subplot(lower[i])
        frame(ax, moved, [(-0.34, 0.34), (-0.09, 0.22)])
        ax.set_title(title, fontsize=7.3, pad=3)
        ax.text(
            0.5,
            -0.16,
            f"{compression[i][step]:.2f} mm  ·  {times[step]:.2f} ms",
            ha="center",
            fontsize=6.8,
            transform=ax.transAxes,
        )
    ax = fig.add_subplot(lower[2])
    for values, color, label in zip(
        compression, ("#187c88", "#b88151"), ("Assigned", "0.25× E"), strict=True
    ):
        ax.plot(times, values, color=color, lw=1.15, label=label)
    ax.legend(loc="upper right", fontsize=6, frameon=False)
    ax.set(xlabel="Time since contact (ms)", ylabel="Compression (mm)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.4)
    fig.text(
        0.06,
        0.034,
        "Same reconstructed mesh and release conditions  ·  Displayed displacement at physical scale",
        size=6.7,
        color="#4f6574",
    )
    for fmt in ("png", "pdf", "svg"):
        path = output / f"cushion_drop_storyboard.{fmt}"
        fig.savefig(path, dpi=360)
        if fmt == "svg":
            path.write_text(
                "\n".join(
                    line.rstrip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                )
                + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("mesh", "particles", "comparison", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--storyboard-only", action="store_true")
    args = parser.parse_args()
    render(
        args.mesh,
        args.particles,
        args.comparison,
        args.output,
        storyboard_only=args.storyboard_only,
    )
