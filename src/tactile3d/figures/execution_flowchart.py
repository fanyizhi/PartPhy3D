"""Operational input/process/output diagram for the target research pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle

from tactile3d.utils.io import ensure_dir, write_json


def generate_execution_flowchart(output_dir: str | Path) -> dict:
    names = {font.name for font in font_manager.fontManager.ttflist}
    font = next((n for n in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei") if n in names), None)
    if font is None:
        raise RuntimeError("Install a Chinese font: Microsoft YaHei or Noto Sans CJK SC")
    plt.rcParams.update({"font.family": font, "pdf.fonttype": 42, "svg.fonttype": "none"})
    out = ensure_dir(output_dir)
    fig = plt.figure(figsize=(11.7, 14), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    checked = []

    def label(x, y, content, size=10, **kwargs):
        return ax.text(x, y, content, transform=ax.transAxes, fontsize=size,
                       color=kwargs.pop("color", "#26333D"), **kwargs)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=12, color="#687983", linewidth=1.2))

    def stage(y, title, lines, files, *, fill="#F2F6F8", edge="#8EA4B2"):
        x, w, h = 0.08, 0.84, 0.070
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=1.2))
        ax.plot([0.665, 0.665], [y+0.009, y+h-0.009], color="#CCD4DA", linewidth=0.8)
        artists = [label(0.096, y+h-0.010, title, 11.5, weight="bold", va="top")]
        for i, line in enumerate(lines):
            artists.append(label(0.096, y+h-0.034-i*0.017, line, 9.1, va="top"))
        artists.append(label(0.682, y+h-0.012, "输出", 8.4, color="#637681", va="top"))
        for i, line in enumerate(files):
            artists.append(label(0.682, y+h-0.032-i*0.017, line, 9, va="top"))
        checked.append(((x, y, w, h), artists))

    label(0.08, 0.973, "PartPhy3D：一个物品从输入到结果", 21, weight="bold", va="top")
    label(0.08, 0.938, "先输入真实图片；再获得完整 Mesh；随后分割、推断和积分；最后生成结果图表。", 10.5)
    stage(0.845, "① 准备输入图片与物品信息", [
        "选择一个物品，整理真实多视角、相机参数和尺度信息。",
        "图片不足或相机未知时，记录缺失并估计；真实质量留作独立评价。",
    ], ["images/、相机与尺度", "独立评价标签"], fill="#EAF2F8")

    for x, title, lines in (
        (0.08, "② A：已有数据集 Mesh", ["直接读取官方参考 GLB / OBJ / PLY。", "跳过重建，先验证分割与物理模块。"]),
        (0.54, "② B：从图片重建 Mesh", ["调用 InstantMesh 等重建方法，先得到完整物体。", "补充尺度标定；生成视角不算新增真实观测。"]),
    ):
        ax.add_patch(Rectangle((x, 0.727), 0.38, 0.080, facecolor="#EFF6F1", edgecolor="#559380", linewidth=1.2))
        artists = [label(x+0.014, 0.793, title, 11, weight="bold", va="top")]
        for i, line in enumerate(lines):
            artists.append(label(x+0.014, 0.765-i*0.018, line, 9, va="top"))
        checked.append(((x, 0.727, 0.38, 0.080), artists))
    ax.plot([0.27, 0.73], [0.825, 0.825], color="#687983", linewidth=1.2)
    ax.plot([0.5, 0.5], [0.845, 0.825], color="#687983", linewidth=1.2)
    arrow(0.27, 0.825, 0.27, 0.809)
    arrow(0.73, 0.825, 0.73, 0.809)
    ax.plot([0.27, 0.27], [0.727, 0.712], color="#687983", linewidth=1.2)
    ax.plot([0.73, 0.73], [0.727, 0.712], color="#687983", linewidth=1.2)
    ax.plot([0.27, 0.73], [0.712, 0.712], color="#687983", linewidth=1.2)
    arrow(0.5, 0.712, 0.5, 0.707)

    stages = [
        (0.635, "③ 检查完整 Mesh，建立图像—表面对应", [
            "统一米制尺度，检查法向、孔洞与连通关系；保留高精度原件。",
            "使用真实视角；必要时另渲染 RGB、深度和法向，记录已知相机。",
        ], ["object_dense.glb", "views/、对应与诊断"]),
        (0.545, "④ 分割完整物体，而非重新生成每个部件", [
            "几何 / 图像 / 部件特征提供候选，优化局部边界与跨视角一致性。",
            "得到逐面标签：部件 → 材料区 → 接缝、接触和连接候选区。",
        ], ["face_labels.npy", "parts.json、掩码图"]),
        (0.455, "⑤ 建立统一状态，推断材料与内部构造", [
            "Part Token＋局部物理场＋连接图，融合不同来源的证据。",
            "推断表面/内部材料、壁厚、骨架、填充和空腔的联合分布。",
        ], ["part_tokens.json", "construction.json"]),
        (0.365, "⑥ 建立内部体积域与材料占用场", [
            "诊断/修补包络，在共享体积域内分配材料与空腔占用率。",
            "表面分割片通常不是封闭实体；独立部件补全需额外验证。",
        ], ["volume_state.npz", "体积与修补诊断"]),
        (0.275, "⑦ 积分计算物理量与不确定性", [
            "由局部密度与占用率计算部件/总质量、质心和完整惯量张量。",
            "输出参数分布；E、ν等需单独监督或适用力学模型。",
        ], ["prediction.json", "part_parameters.csv"]),
        (0.185, "⑧ 可选分支：减面、编辑或仿真导出", [
            "减面与映射保留关键边界和质量矩；编辑后重算构造与物理量。",
            "若仿真需要独立零件，补齐封闭性并检查重叠、间隙和接口。",
        ], ["object_lod.ply", "编辑/仿真资产与报告"]),
        (0.095, "⑨ 独立评价，生成最终图片和表格", [
            "属性图、内部剖面、连接近景；真实/预测质量对比与区间。",
            "输出分割、物理、校准、效率和消融结果；测试真值不反馈推断。",
        ], ["PNG / SVG / PDF", "CSV指标与JSON报告"]),
    ]
    for i, (y, title, lines, files) in enumerate(stages):
        stage(y, title, lines, files,
              fill="#EAF4EF" if i in (1, 2) else "#F2F6F8",
              edge="#559380" if i in (1, 2) else "#8EA4B2")
        if i < len(stages)-1:
            arrow(0.5, y-0.002, 0.5, stages[i+1][0]+0.073)
    arrow(0.5, 0.712, 0.5, 0.708)
    label(0.08, 0.056, "执行关系：先有完整 Mesh，再有部件标签；按需补全独立部件，不必把每个部件从零生成。", 9.2, weight="bold")
    label(0.08, 0.032, "这是目标执行流程：当前原型走②A；学习式精细分割、内部预测与力学验证仍待实现。", 9, color="#62717B")
    fig.canvas.draw()
    inverse = ax.transAxes.inverted()
    renderer = fig.canvas.get_renderer()
    overflow = []
    for (x, y, width, height), artists in checked:
        for artist in artists:
            bounds = artist.get_window_extent(renderer).transformed(inverse)
            if bounds.x0 < x or bounds.x1 > x+width or bounds.y0 < y or bounds.y1 > y+height:
                overflow.append(artist.get_text())
    if overflow:
        plt.close(fig)
        raise RuntimeError(f"Text overflow: {overflow}")
    files = []
    for ext in ("svg", "pdf", "png"):
        path = out / f"partphy3d_execution_flow_zh.{ext}"
        fig.savefig(path, dpi=240, facecolor="white")
        files.append(str(path.resolve()))
    plt.close(fig)
    report = {"artifact": "target processing order and outputs, not completed experiments",
              "text_overflow": overflow, "files": files}
    write_json(out / "execution_flow_qa.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/research_logic")
    args = parser.parse_args()
    print(generate_execution_flowchart(args.output))
