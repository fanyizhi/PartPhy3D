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
    fig, axes = plt.subplots(
        2,
        6,
        figsize=(10.5, 3.7),
        layout="constrained",
        gridspec_kw={"height_ratios": (1, 0.7)},
    )
    for j, rank in enumerate(np.rint(np.linspace(0, len(ids) - 1, 6)).astype(int)):
        scene = ranked[rank]
        path = dataset / "scenes" / scene / "images" / f"{scene}_08.png"
        with Image.open(path) as source:
            rgb = source.convert("RGBA")
            bg = Image.new("RGBA", rgb.size, "white")
            bg.alpha_composite(rgb)
            axes[0, j].imshow(bg.convert("RGB"))
        axes[0, j].set_title(scene.split("_")[0], fontsize=9)
        axes[0, j].axis("off")
        ax = axes[1, j]
        values = [truth[scene], baseline[scene], ours[scene]["updated"]]
        ax.bar(range(3), values, color=["#333333", "#a0abb5", "#4679a1"])
        ax.set_ylim(0, max(values) * 1.3)
        ax.set_xticks(range(3), ["GT", "N2P", "Ours"], fontsize=8)
        for k, v in enumerate(values):
            ax.text(k, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if j == 0:
            ax.set_ylabel("Mass (kg)")
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"test_examples.{ext}", dpi=240)
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
    control = {i: r["mass_kg"] for i, r in sealed(ROOT / "control").items()}
    # Keep all test controls on the original device and frozen evaluation protocol.
    control.update({r["scene_id"]: r["image_prior_kg"] for r in raw})
    ours = sealed(ROOT) if (ROOT / "prediction_seal.json").exists() else test
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

        def measured(label, values):
            m = mass_metrics(truth, values)
            row(
                label,
                "同对象逐件评分",
                [m["ade_kg"], m["alde"], 100 * m["ape"], m["mnre"]],
            )

        if present == len(ids):
            measured("Ours，完整方法", [ours[i]["updated"] for i in ids])
        else:
            lines.append(
                f"| Ours，完整方法 | 已汇入 {present}/{len(ids)}，等待整组封存 | — | — | — | — |"
            )
        measured("仅图像先验（消融）", [control[i] for i in ids])
        if split == "test":
            measured("NeRF2Physics 官方缓存", [baseline[i] for i in ids])
        for name, scope, *vals in PAPER.get(split, []):
            row(
                f"[{name}]({LINKS[name] if name != 'NeRF2Physics' else LINKS['PUGS']})",
                scope,
                vals,
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
    show_test_examples(dataset, splits["test"], y, ours, baseline)
    lines.extend(
        [
            "",
            "训练与验证表缺少五种基线在相同对象上的完整预测。论文原表按作者的数据范围列在表 3、4。物理输入、训练监督和测试对象的接触历史影响跨论文比较。",
            "",
            "![逐对象质量，横纵均使用对数坐标；虚线为等值线](split_mass.png)",
            "",
            "测试集按目录质量排序，展示第 1、21、41、60、80、100 名。所有对象使用第 08 帧，依次比较目录质量、NeRF2Physics 缓存和 Ours。各面板的纵轴随物体质量范围变化，数值单位均为 kg。",
            "",
            "![测试集六个预先按质量秩选择的实物输入和质量估计](test_examples.png)",
            "",
            "RGB 来源为 Amazon Berkeley Objects，Collins 等，CVPR 2022，CC BY 4.0。使用原帧白底合成。[数据来源](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)",
            "",
            "## 材质、物性与模拟图",
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
