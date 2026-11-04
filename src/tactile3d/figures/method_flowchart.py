"""Chinese research-design flowchart, explicitly distinct from measured results."""

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle

from tactile3d.utils.io import ensure_dir, write_json


def generate_method_flowchart(output_dir: str | Path, font_path: str | None = None) -> dict:
    if font_path:
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
    else:
        available = {font.name for font in font_manager.fontManager.ttflist}
        candidates = ("Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei")
        font_name = next((name for name in candidates if name in available), None)
        if font_name is None:
            raise RuntimeError("A Chinese font is required; provide --font /path/to/font.ttf")
    plt.rcParams.update({"font.family": font_name, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.unicode_minus": False})
    out = ensure_dir(output_dir)
    fig = plt.figure(figsize=(15, 12), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    boundaries = []
    ink, muted = "#25313A", "#5B6670"
    blue, teal, orange = "#567AA2", "#3D8C7D", "#B76842"

    def text(x, y, value, size=10, **kwargs):
        return ax.text(x, y, value, fontsize=size, color=kwargs.pop("color", ink),
                       transform=ax.transAxes, **kwargs)

    def box(x, y, w, h, title, lines, *, fill="#F3F6F8", edge="#9CA9B3", tag=None):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=1.2))
        title_artist = text(x+0.012, y+h-0.021, title, 10.6, weight="bold", va="top")
        artists = [title_artist]
        spacing = 0.024 if h > 0.13 else 0.021
        for i, line in enumerate(lines):
            artists.append(text(x+0.012, y+h-0.054-i*spacing, line, 9.1, va="top"))
        if tag:
            artists.append(text(x+0.012, y+0.012, tag, 8.2, color=edge, va="bottom", weight="bold"))
        boundaries.append(((x, y, w, h), artists, title))

    def arrow(points, *, color="#6A7780", dashed=False, width=1.4):
        for start, end in pairwise(points[:-1]):
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color,
                    linewidth=width, linestyle="--" if dashed else "-", zorder=0)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                     mutation_scale=12, linewidth=width, color=color,
                                     linestyle="--" if dashed else "-", zorder=0))

    text(0.04, 0.963, "PartPhy3D：准确物理感知、部件编辑与增量更新", 21, weight="bold", va="top")
    text(0.04, 0.925, "IEEE Transactions 研究方案｜物理准确性与可编辑性共同验证，以独立机械臂抓取和搬运检验操作作用", 10.6, color=muted)

    xs = (0.04, 0.28, 0.52, 0.76)
    w, h = 0.20, 0.165
    problems = (
        ("问题 P1：物理属性预测不准", "尺度与隐藏构造影响质量分布"),
        ("问题 P2：编辑不同步", "材料变化后结构与接口未更新"),
        ("问题 P3：整物体反复重算", "缺少身份、依赖与缓存复用"),
        ("问题 P4：操作作用缺少证据", "需要独立测量和固定控制对照"),
    )
    for x, (title, body) in zip(xs, problems, strict=True):
        ax.add_patch(Rectangle((x, 0.847), w, 0.049, facecolor="#F6F6F6", edgecolor="#C6CDD2", linewidth=0.8))
        text(x+0.010, 0.886, title, 9.5, weight="bold", va="top")
        text(x+0.010, 0.863, body, 8.2, va="top", color=muted)

    y1, y2 = 0.638, 0.397
    box(xs[0], y1, w, h, "① 数据与证据", [
        "数据集 RGB、相机与尺度",
        "ABO-500：统一质量测试协议",
        "局部测量：材料、摩擦与刚度",
        "已知构造与实物：编辑后参考",
    ], tag="按物体划分训练 / 验证 / 测试")
    box(xs[1], y1, w, h, "② 高精度几何与空间对应", [
        "参考 Mesh 与重建 Mesh 分轨",
        "InstantMesh 作为重建候选",
        "尺度、法向、孔洞与拓扑诊断",
        "相机投影、遮挡检查、表面锚点",
    ], tag="保留原始细节，先分割再减面")
    box(xs[2], y1, w, h, "③ 精细分割与构造推断", [
        "整物体 → 部件 → 局部区域",
        "几何 / 图像 / Qwen 构造候选",
        "跨视角对应与内部占用分布",
        "质量及可信局部属性联合监督",
    ], fill="#EAF4F0", edge=teal, tag="C1：构造感知的准确物理预测")
    box(xs[3], y1, w, h, "④ 统一可编辑部件状态", [
        "Part Token：身份、语义、证据",
        "Local Field：局部材料与占用率",
        "Connection Graph：连接与依赖",
        "积分质量、COM 与惯量分布",
    ], fill="#EAF4F0", edge=teal, tag="C1：推断状态直接用于编辑")
    for i in range(3):
        arrow([(xs[i]+w+0.003, y1+h/2), (xs[i+1]-0.004, y1+h/2)])
    arrow([(xs[3]+w/2, y1-0.003), (xs[3]+w/2, y2+h+0.003)])
    text(xs[3]+w/2-0.018, 0.597, "用户选择部件并发出编辑指令", 8.3, ha="right", color=muted)

    box(xs[3], y2, w, h, "⑤ 解析并执行部件编辑", [
        "换材料 / 改内部 / 替换几何",
        "EditSpec：目标、接口、保护区",
        "构造与物理分布同步更新",
        "保留未编辑部件及有效状态",
    ], fill="#FFF2E8", edge=orange, tag="C2：材料—结构—几何协同编辑")
    box(xs[2], y2, w, h, "⑥ 接口约束与局部适配", [
        "状态更新 → 局部壳厚与占用",
        "必要时检索 / 生成 donor 部件",
        "接口对齐、碰撞与保护区约束",
        "按显式代价选择可行候选",
    ], fill="#FFF2E8", edge=orange, tag="C2：仅在约束需要时改几何")
    box(xs[1], y2, w, h, "⑦ 依赖驱动的增量更新", [
        "受影响子图与缓存失效传播",
        "仅重算变化字段、接口与质量矩",
        "更新质量、COM、完整惯量分布",
        "非局部依赖触发扩大重算范围",
    ], fill="#EDF2F8", edge=blue, tag="C3：精度与更新成本共同验证")
    box(xs[0], y2, w, h, "⑧ 分级验证与编辑提交", [
        "几何 / 占用 / 解析物理检查",
        "有对应数据时验证力学响应",
        "失败返回违反项或升级适配",
        "成功提交资产、记录与可撤销版本",
    ], tag="模型内通过不等于实物强度认证")
    for i in (3, 2, 1):
        arrow([(xs[i]-0.004, y2+h/2), (xs[i-1]+w+0.004, y2+h/2)])
    arrow([(0.62, y2+h+0.002), (0.62, y1-0.004)], color=orange, dashed=True, width=1.5)
    text(0.60, 0.613, "新部件 / 高风险接口 → 局部边界细化", 8.1, ha="right", color=orange)
    text(0.28, 0.366, "最终结果与论文证据", 12.2, weight="bold")

    output_y, output_h = 0.190, 0.124
    outputs = (
        ("结果 1：物理准确性主实验", ["统一质量 ADE / ALDE / MnRE 等",
                                  "局部密度、摩擦、刚度与校准",
                                  "PUGS / N2P / PhysVGGT 对照"]),
        ("结果 2：部件编辑主实验", ["材料 / 构造 / 几何编辑前后",
                                  "接口误差、保护区漂移、成功率",
                                  "质量、COM、惯量变化与误差"]),
        ("结果 3：增量更新主实验", ["与同条件全量重算比较精度",
                                  "耗时、显存、缓存与更新范围",
                                  "随部件数 / 编辑范围变化曲线"]),
        ("结果 4：独立机械臂实验", ["固定控制策略，仅替换属性输入",
                                  "成功率、滑移、接触力与形变",
                                  "编辑前后配对；报告失败与拒绝"]),
    )
    for x, (title, lines) in zip(xs, outputs, strict=True):
        box(x, output_y, w, output_h, title, lines, fill="#F3F6FA", edge=blue)
    arrow([(xs[0]+w/2, y2-0.003), (xs[0]+w/2, 0.338), (xs[0]+w/2, output_y+output_h+0.003)])
    ax.plot([xs[0]+w/2, xs[3]+w/2], [0.338, 0.338], color="#6A7780", linewidth=1.0, zorder=0)
    for x in xs[1:]:
        arrow([(x+w/2, 0.338), (x+w/2, output_y+output_h+0.003)], width=1.0)

    text(0.04, 0.143, "已有原型", 9.5, weight="bold", color=teal)
    text(0.12, 0.143, "官方 ABO 资产、几何分组、显式物理状态、donor 替换与基础刚体属性重构、属性图及连接几何检查。", 9)
    text(0.04, 0.114, "待实现验证", 9.5, weight="bold", color=orange)
    text(0.12, 0.114, "质量 / 局部属性学习与统一基准、构造与接口适配、版本依赖及局部缓存、独立编辑与机械臂验证。", 9)
    text(0.04, 0.066, "解释边界：生成视角不等于真实观测；连通片不等于制造零件；总质量正确不等于内部正确；表面接近不等于连接牢固。", 8.5, color=muted)
    text(0.04, 0.039, "绿色 C1 / 橙色 C2 / 蓝色 C3 为拟议贡献，需以公平基线和消融确立；图示目标方法，不表示完整系统已实现。", 8.5, color=muted)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = ax.transAxes.inverted()
    overflow = []
    for (x, y, width, height), artists, title in boundaries:
        for artist in artists:
            bounds = artist.get_window_extent(renderer).transformed(inverse)
            if (bounds.x0 < x or bounds.x1 > x+width or bounds.y0 < y
                    or bounds.y1 > y+height):
                overflow.append({"box": title, "text": artist.get_text()})
    if overflow:
        plt.close(fig)
        raise RuntimeError(f"Text exceeds a box: {overflow}")
    paths = []
    for suffix in ("svg", "pdf", "png"):
        path = out / f"partphy3d_research_logic_zh.{suffix}"
        fig.savefig(path, dpi=240, facecolor="white")
        paths.append(str(path.resolve()))
    plt.close(fig)
    report = {"artifact": "target research design, not an experiment result",
              "font": font_name, "box_count": len(boundaries), "text_overflow": overflow,
              "figure_files": paths}
    write_json(out / "diagram_qa.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/research_logic")
    parser.add_argument("--font")
    args = parser.parse_args()
    print(generate_method_flowchart(args.output, args.font))
