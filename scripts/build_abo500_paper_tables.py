"""Build the four ABO mass tables and split-wise prediction figure from sealed outputs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from prepare_abo50_benchmark import digest, read_json
from run_abo100_role_replay import BASE, OUT
from tactile3d.models.calibration import mass_metrics

ROOT = BASE.parent / "abo500_20261120"
LINKS = {
    "NeRF2Physics": "https://openaccess.thecvf.com/content/CVPR2024/html/Zhai_Physical_Property_Understanding_from_Language-Embedded_Feature_Fields_CVPR_2024_paper.html",
    "PUGS": "https://marcyu0303.github.io/assets/pdfs/gs_physics.pdf#page=5",
    "PhysVGGT": "https://arxiv.org/html/2609.18920v1#A1.T13",
    "GS2Physics": "https://ieeexplore.ieee.org/document/11653536/",
    "Gen2Physics": "https://arxiv.org/html/2608.23869v1#S4.T1",
}
# Published, protocol-specific aggregates. APE is already converted to percent.
PAPER = {
    "test": [
        ("PhysVGGT", "原文表 13，单图全局头；作者测试集", 4.730, 0.574, 78.7, 0.655)
    ],
    "all": [
        ("NeRF2Physics", "PUGS 表 I 报告的对照", 12.725, 0.736, 104.0, 0.564),
        ("PUGS", "原文表 I，全 500 件", 9.461, 0.661, 76.7, 0.576),
        ("GS2Physics", "原文表 I，Test1–3 共 500 件", 9.884, 0.720, 88.8, 0.559),
        ("Gen2Physics", "原文表 1，ABO-500，参考资产输入", 8.447, 0.651, 117, 0.600),
    ],
}


def sealed(directory):
    path = directory / "predictions.json"
    if digest(path) != read_json(directory / "prediction_seal.json")["sha256"]:
        raise ValueError(f"Invalid prediction seal in {directory}")
    return read_json(path)


def show_test_examples(dataset, ids, truth, ours, baseline):
    """Choose six mass ranks before comparing errors and use one shared camera view."""
    ranked = sorted(ids, key=lambda i: (truth[i], i))
    chosen = [ranked[j] for j in np.rint(np.linspace(0, len(ids) - 1, 6)).astype(int)]
    fig = plt.figure(figsize=(7.1, 5.2), layout="constrained")
    grid = fig.add_gridspec(6, 2, width_ratios=(1, 5))
    ax = fig.add_subplot(grid[:, 1])
    series = [
        ("Catalogue", lambda i: truth[i], "#30363b", "D", .19),
        ("NeRF2Physics", lambda i: baseline[i], "#856aa1", "s", 0.),
        ("Ours", lambda i: ours[i]["updated"], "#15796f", "o", -.19),
    ]
    for j, scene in enumerate(chosen):
        photo = fig.add_subplot(grid[j, 0])
        with Image.open(dataset / "scenes" / scene / "images" / f"{scene}_08.png") as source:
            rgba = source.convert("RGBA")
            bounds = rgba.getchannel("A").getbbox()
            rgba = rgba.crop(bounds) if bounds else rgba
            white = Image.new("RGBA", rgba.size, "white")
            white.alpha_composite(rgba)
            photo.imshow(white.convert("RGB"))
        photo.set_axis_off()
    for label, mass, color, marker, offset in series:
        ax.scatter([mass(i) for i in chosen], np.arange(6) + offset,
                   color=color, marker=marker, s=26, label=label, zorder=3)
    ax.set(xscale="log", ylim=(5.5, -.5), xlabel="Mass (kg, logarithmic scale)")
    ax.set_yticks(range(6), [s.split("_")[0] for s in chosen], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#dedede", linewidth=.6)
    for j in range(5):
        ax.axhline(j + .5, color="#eeeeee", linewidth=.6)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"test_examples.{ext}", dpi=300)
    plt.close(fig)


def show_benchmark_comparison(groups):
    """Display measured and published values with a common metric scale."""
    panels = ("val", "test", "all")
    heights = [len(groups[split]) + 1 for split in panels]
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(13, 7.5),
        sharex="col",
        layout="constrained",
        gridspec_kw={"height_ratios": heights},
    )
    for row, split in enumerate(panels):
        entries = groups[split]
        labels = [entry[0] for entry in entries]
        ypos = np.arange(len(labels))
        for col, (metric, precision) in enumerate(
            (("ADE (kg) ↓", 2), ("ALDE ↓", 3), ("APE (%) ↓", 1), ("MnRE ↑", 3))
        ):
            ax = axes[row, col]
            values = [entry[1][col] for entry in entries]
            for pos, (label, _, published), value in zip(ypos, entries, values):
                color = "#276aa0" if label == "Ours" else "#65727d"
                ax.scatter(
                    value,
                    pos,
                    s=30,
                    facecolor="white" if published else color,
                    edgecolor=color,
                    zorder=3,
                )
                ax.annotate(
                    f"{value:.{precision}f}",
                    (value, pos),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                )
            ax.set_yticks(ypos, labels if col == 0 else [""] * len(labels))
            ax.set_ylim(len(labels) - 0.45, -0.55)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", color="#e4e9ed", linewidth=0.7)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.margins(x=0.22)
            if row == 0:
                ax.set_title(metric)
        axes[row, 0].set_ylabel(
            {"val": "Validation · 100", "test": "Test · 100", "all": "Full benchmark"}[
                split
            ],
            fontweight="semibold",
        )
    fig.text(
        0.5,
        1.005,
        "Filled markers: this benchmark · Open markers: values reported in papers",
        ha="center",
        fontsize=10,
    )
    with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        for ext in ("svg", "pdf", "png"):
            fig.savefig(
                ROOT / f"mass_baseline_comparison.{ext}", dpi=240, bbox_inches="tight"
            )
    svg = ROOT / "mass_baseline_comparison.svg"
    svg.write_bytes(
        b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines()) + b"\n"
    )
    plt.close(fig)


def build(dataset, weights):
    splits = read_json(ROOT / "splits.json")
    raw = read_json(BASE / "final100/score.json")["per_scene"]
    test = read_json(OUT / "predictions.json")
    if (
        digest(OUT / "predictions.json")
        != read_json(OUT / "metrics.json")["prediction_sha256"]
    ):
        raise ValueError("Archived test prediction seal mismatch")
    if set(test) != set(splits["test"]):
        raise ValueError("Archived test cohort mismatch")
    baseline = {r["scene_id"]: r["nerf_mass_kg"] for r in raw}
    if (ROOT / "nerf_run/prediction_seal.json").is_file():
        intervals = sealed(ROOT / "nerf_run")
        midpoint = {i: sum(v) / 2 for i, v in intervals.items()}
        if any(midpoint[i] != value for i, value in baseline.items()):
            raise ValueError("Archived NeRF2Physics test results changed")
        baseline = midpoint
    control = {i: r["mass_kg"] for i, r in sealed(ROOT / "control").items()}
    # Keep all test controls on the original device and frozen evaluation protocol.
    control.update({r["scene_id"]: r["image_prior_kg"] for r in raw})
    ours = dict(test)
    for batch in sorted((ROOT / "batches").glob("*")):
        split, start = batch.name.split("_")
        chunk = sealed(batch)
        ids = [row["scene_id"] for row in chunk]
        if ids != splits[split][int(start) : int(start) + 25] or set(ids) & ours.keys():
            raise ValueError(f"Invalid or overlapping sealed batch {batch.name}")
        ours.update(zip(ids, chunk))
    if (ROOT / "prediction_seal.json").exists():
        final = sealed(ROOT)
        if set(final) != set(sum(splits.values(), [])) or any(
            final[i]["updated"] != r["updated"] for i, r in ours.items()
        ):
            raise ValueError("Full record differs from earlier sealed batches")
        ours = final
    if not all(ours[i]["updated"] == test[i]["updated"] for i in test):
        raise ValueError("Frozen test predictions changed")
    all_ids = splits["train"] + splits["val"] + splits["test"]
    if len(set(all_ids)) != 500 or set(all_ids) != set(control):
        raise ValueError("Control cohort mismatch")
    groups = dict(splits, all=all_ids)
    y = {i: weights.get(i, weights.get(i.split("_")[0])) for i in all_ids}
    names = dict(train="训练集", val="验证集", test="测试集", all="全体对象")
    lines = [
        "# ABO-500 质量估计",
        "",
        "图像质量头在官方训练 300 件拟合，在验证 100 件选择正则系数。六视角重建及三视角物理推理沿用冻结设置。测试 100 件曾用于诊断。全体汇总含训练对象。",
        "",
        f"完整方法已纳入本页的封存预测为 {len(ours)}/500 件。图像先验有 500/500 件，训练及验证使用 CPU 重放冻结头，测试沿用原 GPU 预测。每张表仅在对应范围完整时给出本方法汇总。",
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(9, 3.6), sharex=True, sharey=True, layout="constrained"
    )
    limits = [
        min(
            min(y.values()),
            min(control.values()),
            min(v["updated"] for v in ours.values()),
        )
        * 0.8,
        max(
            max(y.values()),
            max(control.values()),
            max(v["updated"] for v in ours.values()),
        )
        * 1.2,
    ]
    comparison = {split: [] for split in groups}
    for number, (split, ids) in enumerate(groups.items(), 1):
        truth = [y[i] for i in ids]
        present = sum(i in ours for i in ids)
        lines.extend(
            [
                "",
                f"## 表 {number}　{names[split]}，{len(ids)} 件",
                "",
                "| 方法 | 评测口径 | ADE (kg) ↓ | ALDE ↓ | APE (%) ↓ | MnRE ↑ |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )

        def row(label, scope, vals):
            a, b, c, d = vals
            lines.append(
                f"| {label} | {scope} | {a:.3f} | {b:.4f} | {c:.2f} | {d:.4f} |"
            )

        def measured(label, figure_label, values):
            m = mass_metrics(truth, values)
            scores = [m["ade_kg"], m["alde"], 100 * m["ape"], m["mnre"]]
            row(label, "同对象逐件评分", scores)
            comparison[split].append((figure_label, scores, False))

        if present == len(ids):
            measured("Ours，完整方法", "Ours", [ours[i]["updated"] for i in ids])
        else:
            lines.append(
                f"| Ours，完整方法 | 已汇入 {present}/{len(ids)}，等待整组封存 | — | — | — | — |"
            )
        measured("仅图像先验（消融）", "Image prior", [control[i] for i in ids])
        if all(i in baseline for i in ids):
            measured(
                "NeRF2Physics 官方缓存", "NeRF2Physics", [baseline[i] for i in ids]
            )
        for name, scope, *vals in PAPER.get(split, []):
            row(
                f"[{name}]({LINKS[name] if name != 'NeRF2Physics' else LINKS['PUGS']})",
                scope,
                vals,
            )
            comparison[split].append(
                (
                    "NeRF2Physics (PUGS Table I)" if name == "NeRF2Physics" else name,
                    vals,
                    True,
                )
            )
        if split != "all":
            ax = axes[number - 1]
            ax.loglog(limits, limits, "--", lw=0.8, color=".5")
            ax.scatter(
                truth,
                [control[i] for i in ids],
                s=12,
                alpha=0.6,
                color="#77838f",
                label="Image prior",
            )
            if present == len(ids):
                ax.scatter(
                    truth,
                    [ours[i]["updated"] for i in ids],
                    s=12,
                    alpha=0.65,
                    color="#276aa0",
                    label="Full method",
                )
            ax.set(
                xscale="log",
                yscale="log",
                xlim=limits,
                ylim=limits,
                aspect="equal",
                title=f"{split.title()} (n = {len(ids)})",
                xlabel="Catalogue mass (kg)",
            )
            ax.legend(frameon=False, fontsize=7, loc="upper left")
    axes[0].set_ylabel("Predicted mass (kg)")
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"split_mass.{ext}", dpi=240)
    plt.close(fig)
    show_benchmark_comparison(comparison)
    show_test_examples(dataset, splits["test"], y, ours, baseline)
    lines.extend(
        [
            "",
            "NeRF2Physics 的同对象行使用作者缓存的 30 视角属性阶段，区间取算术中点。论文原表按作者的数据范围列在表 3、4。物理输入、训练监督和测试对象的接触历史影响跨论文比较。",
            "",
            "![验证、测试及全量的四项质量指标对比](mass_baseline_comparison.png)",
            "",
            "实心点为本实验逐对象评分，空心点为各论文报告值。全量面板含图像头训练使用的对象。[质量对比图 PDF](mass_baseline_comparison.pdf) · [可编辑 SVG](mass_baseline_comparison.svg)",
            "",
            "![逐对象质量，横纵均使用对数坐标；虚线为等值线](split_mass.png)",
            "",
            "测试集按目录质量排序，展示第 1、21、41、60、80、100 名。所有对象使用第 08 帧，依次比较目录质量、NeRF2Physics 缓存和 Ours。六行共用一条以 kg 为单位的对数横轴，同一行的三个标记对应同一个物体。",
            "",
            "![测试集六个预先按质量秩选择的实物输入和质量估计](test_examples.png)",
            "",
            "RGB 来源为 Amazon Berkeley Objects，Collins 等，CVPR 2022，CC BY 4.0。使用原帧白底合成。[数据来源](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)",
            "",
            "## 材质、物性与模拟图",
            "",
            "下图对应 PUGS、GS2Physics、Gen2Physics 的材质分区以及 PhysVGGT 的分属性视图。以两把有 PhysSeg 标注的椅子为例，每件固定取按帧编号排序的第 1、3 个标注视图；评测使用全部 10 张。用相同的相机和裁剪排列输入、公开标注、预测材质与按材质赋予的杨氏模量先验。两类 mIoU 为 64.13%（织物 95.55%、木材 32.71%）。蓝椅细腿被预测成钢，公开标注为木材，这个分歧同时影响材质评分和刚度图。该小样本来自原有外观补全诊断。",
            "",
            "![同视角材质和杨氏模量](physseg_material_modulus.png)",
            "",
            "[PhysSeg-15 作者标注](https://github.com/momaiyc/GS2Physics) · [十视角计数](../abo50_coverage_20261119/appearance/physseg_overlap.json) · [材质与模量矢量图](physseg_material_modulus.pdf)",
            "",
            "沙发和木凳图依次展示输入、构造剖面、内部粒子和压缩响应。沙发内部结构采用模型假设，木凳是有构造真值的合成对照，图中位移放大 35 倍。对应 Gen2Physics 的部件与模拟、PUGS 的物性到任务验证。",
            "",
            "![既有两例 MPM 验证](../mesh_interior_mpm_20261119/material_points_mpm.png)",
            "",
            "躺椅参考网格单例按固定视角展示材质及表面物性先验，对照 PhysVGGT 和 GS2Physics 按属性分列的画法。该例使用参考几何。不同量纲分别标尺，未知区域保留在图中。",
            "",
            "![既有单例材质及物性先验](../chaise_reference_20261111_final/dense_property_maps.png)",
            "",
            "NeRF2Physics 与 PUGS 的同视角示例、五篇论文来源、材质与标注的详细口径见[既有逐项对照](../../docs/results/abo50_physical_comparison_20261119_zh.md)。",
            "",
            "[逐对象散点矢量图](split_mass.pdf) · [封存的测试实验](../../docs/results/abo100_role_replay_zh.md)",
        ]
    )
    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    source = parser.parse_args().weights
    build(source.parent, read_json(source))
