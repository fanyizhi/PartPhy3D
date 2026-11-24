# ABO-500 质量对照与材质验证

## Ours 与五篇基线的质量指标

| 方法 | ADE (kg) ↓ | ALDE ↓ | APE (%) ↓ | MnRE ↑ |
|---|---:|---:|---:|---:|
| NeRF2Physics | 8.730 | 0.771 | 106.10 | 0.552 |
| PUGS | 9.461 | 0.661 | 76.70 | 0.576 |
| PhysVGGT | 4.730 | 0.574 | 78.70 | 0.655 |
| GS2Physics | 9.884 | 0.720 | 88.80 | 0.559 |
| Gen2Physics | 8.447 | 0.651 | 117.00 | 0.600 |
| **Ours** | 6.807 | 0.488 | 56.89 | 0.663 |

**表注。** Ours 使用官方训练 300、验证 100、测试 100 的全量 500 件，评分读取每件封存预测的 `updated` 字段。官方测试 100 件单列为 5.696 kg、0.577、70.62%、0.620，该批对象此前用于诊断。NeRF2Physics 和 PhysVGGT 使用原文各自的测试 100 件，后者为 global mass head。PUGS 使用原文全量 500 件，GS2Physics 合并 Test1–3 的 500 件，Gen2Physics 使用 500 件参考资产。各原论文的输入、监督和划分随所报成绩一起解释。 [预测与封存](../abo500_20261120/README.md) · [NeRF2Physics 表 1](https://arxiv.org/html/2404.04242#S4.T1) · [PUGS 表 I](https://marcyu0303.github.io/assets/pdfs/gs_physics.pdf#page=5) · [PhysVGGT 表 13](https://arxiv.org/html/2609.18920v1#A1.T13) · [GS2Physics 表 I](https://ieeexplore.ieee.org/document/11653536/) · [Gen2Physics 表 1](https://arxiv.org/html/2608.23869v1#S4.T1)

ADE 是平均绝对质量误差，ALDE 是平均绝对自然对数误差，APE 是平均绝对百分比误差，MnRE 是逐件取预测与参考质量两种相除方向中的较小值再求平均，越接近 1 越好。

## Ours 的方法与实现

方法说明给出相机选择、网格绕向修复、逐面投票和补全、三类材料体积、结构修正、有界融合的公式与代码。实例实验接着展示互斥材料域、局部换材的质量矩增量，以及内部粒子如何驱动 MLS-MPM。 [完整中文方法与实现](../../docs/PARTPHY3D_METHODS_zh.md) · [编辑与仿真](../part_edit_20261124/README.md)

## 同视角的物体材质对比

![四件 ABO 物体的输入、区域证据以及 Ours、NeRF2Physics 和 MaskCLIP 的材质输出](../part_edit_20261124/material_selected.png)

主图从已有七例的固定第 08 帧中，按 Ours 的前景赋值覆盖选前四。USB 集线器、花纹椅、紫色椅和写字台覆盖分别为 97.29%、96.44%、95.27%、95.20%。每列共享相机和裁剪，区域证据用三视图部件票向量的 PCA 显色，预测行用共享类别色板。选择规则用于定性展示，覆盖率衡量是否完成赋值。 [主图 PDF](../part_edit_20261124/material_selected.pdf) · [逐例相机与选择分数](../part_edit_20261124/material_selected_data.json) · [完整七例](../part_edit_20261124/region_material_seven.pdf)

![两把椅子的 RGB、人工标注、Ours、NeRF2Physics、PUGS–Qwen 和 MaskCLIP](paired_material.png)

| 方法 | Fabric IoU (%) ↑ | Wood IoU (%) ↑ | mIoU (%) ↑ |
|---|---:|---:|---:|
| NeRF2Physics | 96.33 | 34.57 | 65.45 |
| PUGS–Qwen | 81.34 | 0.03 | 40.68 |
| MaskCLIP (2D) | 69.39 | 52.84 | 61.12 |
| **Ours** | **95.55** | **32.71** | **64.13** |

三维方法在两把椅子的五个 PhysSeg 视角分别取公共相机和共享网格。NeRF2Physics 复用作者缓存和官方查询代码，PUGS–Qwen 使用 PUGS 原实现和 Qwen 材料提议，在固定半径内将预测概率转到同一网格。MaskCLIP 是新加的图像基线，用冻结的 ViT-B/16 DataComp 权重、448 px 输入和固定的十二类材料提示，直接分配相同 RGB 视角内的像素，使用 RGB 原有透明通道限制背景。它以原论文的末层 value 特征路径实现，未采用可选的类提示滤波与邻域平滑。

四个方法均在恰有一个人工标注类别的像素上评分。先在十个视角汇总各类交集与并集，再把织物与木材 IoU 等权平均。单看图中的花纹椅视角，Ours 为 87.63%，NeRF2Physics 为 75.12%，PUGS–Qwen 为 44.09%，MaskCLIP 为 73.49%。十视角汇总里，NeRF2Physics 的 mIoU 最高，MaskCLIP 的木材 IoU 最高。 [材料图 PDF](paired_material.pdf) · [三维方法逐视角计数](paired_material.json) · [MaskCLIP 逐视角计数](maskclip_physseg.json) · [模型和输入记录](maskclip_manifest.json) · [MaskCLIP 原论文与代码](https://github.com/chongzhou96/MaskCLIP)

### 其他基线的原文材质结果

| 方法 | Gen2Physics 表 2，PartNet-Material mIoU (%) | GS2Physics 表 II，PhysSeg-15 Eval1 / Eval2 / Eval3 mIoU (%) |
|---|---:|---:|
| NeRF2Physics | 14.54 | 48.5 / 31.1 / 41.2 |
| PUGS | 15.56 | 18.7 / 27.9 / 29.3 |
| GS2Physics | — | 68.9 / 58.7 / 81.7 |
| Gen2Physics | 48.27 | — |
| PhysVGGT | — | — |

GS2Physics 已发布 PhysSeg-15 人工标注和模拟结果，材质预测图见原文 Fig. 8。Gen2Physics 材质图见原文 Fig. 4，样本来自 PartNet-Material。PhysVGGT 的 Fig. 4 是密度、摩擦、硬度和杨氏模量预测。当前公开资料中未找到三个方法在上述同视角 ABO 实例上的可复算预测。原论文图与成绩按各自数据和任务引用。 [GS2Physics Fig. 8 和表 II](https://ieeexplore.ieee.org/document/11653536/) · [作者仓库](https://github.com/momaiyc/GS2Physics) · [Gen2Physics Fig. 4](https://arxiv.org/html/2608.23869v1#S4.F4) · [PhysVGGT Fig. 4](https://arxiv.org/html/2609.18920v1#S4.F4)

## 本轮发现的问题与修正验证

- **材质基线在公开权重和输出格式上有差异。** CLIPSeg 权重下载端点从实验服务器连接失败。改用已有已封存的 DataComp ViT-B/16 权重实现 MaskCLIP 的末层 value 路径，在服务器为所展示案例推理，并用 PhysSeg 标注的同十个视角核算。渲染时比对输入与输出 SHA-256。新增基线的木材 IoU 达 52.84%，下一轮可据此检查 2D 细结构证据在哪些视角补回了三维投影遗漏。 [基线实现](../../scripts/run_maskclip_material.py) · [实验记录](maskclip_manifest.json)
- **材质与几何有不同的失效原因。** 蓝椅的 8,900 个木材标注像素中，4,563 个看不到网格，3,844 个被赋成金属。剩下 493 个为无标签或其他材料，正确木材像素为 0。把冻结的亚像素轮廓网格投影到同五个留出视角，木材标注像素的几何命中从 **48.73% 提到 78.38%**，整件前景精度从 **96.86% 到 96.97%**。花纹椅的四个共同留出视角里，木材几何命中从 84.68% 到 95.53%，前景精度从 97.57% 降至 92.57%。因此本轮保留原网格的材质主结果。局部细件恢复具备潜力，仍需按部件转移材料证据并验证边界。 [逐视角几何和材质计数](paired_material.json)
- **质量验证集上的损失集中在错误方向的结构修正。** 对 100 件验证物体，用每件实际进入融合的图像质量作同对象对照。完整融合有 48 件误差变小、52 件变大，净增 0.932 kg ADE。`B07QHYD94M_ATVPDKIKX0DER` 的目录质量为 55.34 kg，图像支路为 107.84 kg，构造修正把结果推到 151.63 kg，单件新增 43.79 kg 绝对误差。单靠改变最终的融合权重无法识别这类异常，需约束输入物理支路的尺度、实体占用和材料密度。主表继续使用封存完整方法。 [结构质量修正对照](../abo500_20261120/fusion_validation/README.md) · [验证失败归因](../abo500_20261120/validation_failure_audit.json)

RGB 来自 Collins 等的 Amazon Berkeley Objects，CC BY 4.0。图像统一裁剪、缩放和白底合成。 [ABO 数据](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
