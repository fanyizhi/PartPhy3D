# PartPhy3D 方案固定版

> 固定日期：2026-09-15（昨晚讨论内容）  
> 状态：作为当前项目的研究与实现基线。后续修改应另写版本记录，不直接覆盖本文件。

## 1. 项目目标

项目名称为 **PartPhy3D**。目标是构建一个运行于 Linux 的完整论文代码项目，以显式
mesh 替代 NeRF2Physics 和 PUGS 使用的隐式场或 3D Gaussian 表示，实现：

1. 从单张或多视角 RGB 图像获得物体 mesh，或者直接输入已有 textured mesh。
2. 对 mesh 做部件级分割。
3. 对每个部件推断表面材料和不可见内部结构。
4. 输出密度、杨氏模量、质量等物理属性的概率分布。
5. 对水密体积、壳体、泡沫、框架、填充物和空腔进行显式质量积分。
6. 融合多视角证据，降低遮挡、反光、模糊和单视角歧义造成的偏差。
7. 支持部件级替换，并在替换后重新计算质量、质心、稳定性和接口质量。

严格表述：只有图像和 mesh 时，系统得到的是**视觉条件下的触觉/物理属性先验**。没有
力-位移、GelSight、DIGIT 或材料实验数据时，不能宣称已经观测到真实触觉。

## 2. 数据与基线

### 主数据集

- **ABO-500**：500 个物体、每个物体 30 个视角，固定 300/100/100 训练、验证、测试划分。
- 原始 ABO 还提供 7,900 多个带 PBR 纹理的 glTF/GLB mesh；本项目可直接读取 GLB，但必须
  通过 `3dmodel_id` 核验其与 ABO-500 场景的对应关系、相机坐标和真实尺度。
- ABO-500 提供物体级真实质量，但不提供可靠的部件密度、杨氏模量和内部结构真值。
- 因此它适合验证总质量，不足以独立证明部件级物理属性准确。

### 推荐增加的数据

- 材料密度与杨氏模量测量表，记录测试标准、温度、方向和不确定度。
- 产品 BOM、CAD 或 CT，用于监督隐藏内部结构。
- PartNet/PartNet-Mobility，用于语义部件和 donor part，但不能当作质量真值。
- 专家标注的表面材料、内部构造候选、壁厚和填充率。
- 经过伦理审批的心理物理成对比较数据，用于“看起来更重/更软/更稳定”的知觉任务。

### 对照方法

- **NeRF2Physics**：多视角 NeRF、语言对齐特征和物理属性体积分。
- **PUGS**：区域/几何感知 3D Gaussian 和 Gaussian 体积估计。
- **PartPhy3D**：显式、可修复、可分割、可编辑的 mesh 加隐藏内部结构后验。

创新点不是简单换输入格式，而是利用 mesh 明确表达水密边界、部件体积、壳层/填充结构、
物理约束和编辑后的可验证变化。

## 3. 完整方法

```text
RGB / multi-view RGB / textured metric mesh
    -> mesh reconstruction or mesh loading
    -> metric-scale recovery
    -> mesh repair and watertightness validation
    -> semantic/geometric part segmentation
    -> deterministic multi-view rendering
    -> per-part surface-material evidence
    -> VLM top-K interior construction hypotheses
    -> voxel/volume allocation to parts
    -> Monte Carlo physical integration
    -> mass, density, modulus and uncertainty
    -> optional physics-aware part replacement
```

VLM 对沙发等复合物体不能只回答“皮革”。每个部件需要输出两层材料：

- 表面：皮革、织物、木材、金属、塑料等。
- 内部：泡沫、棉、木框、金属弹簧、蜂窝、空气或其他填充。

建议让 VLM 一次给出 2-4 个带概率的结构化候选，而不是不断自由问答直到“猜对”。输出包括：

```text
surface_material_probabilities
interior_structure_probabilities
shell_thickness_interval
core/frame/infill_materials
occupied_volume_fraction
evidence and confidence
```

物理约束层负责检查概率归一化、体积分数守恒、非负壁厚、合理密度范围和质量一致性。VLM
只提供先验，不直接决定最终质量。

## 4. 水密 mesh、填充与质量

闭合三角 mesh 的体积可由有向四面体积分计算：

```text
V = abs(sum(dot(v0, cross(v1, v2))) / 6)
```

但该公式要求 mesh 闭合、法向基本一致、尺度已转换为米。正式流程是：

1. 检查非流形边、孔洞、自相交、重复面和法向。
2. 小孔使用 fill-holes 修复；严重缺失不能伪装成精确体积。
3. 体素化内部，以部件表面或最近部件分配内部体素。
4. 使用精确封闭体积校正体素分配误差。
5. 对实心、壳体、分层、泡沫和框架分别积分。

```text
solid:   mass = density * enclosed_volume
hollow:  mass = shell_density * shell_area * thickness
layered: mass = sum(density_j * occupied_volume_j)
object:  mass = sum(part_mass_k)
```

尺度是首要条件。mesh 长度缩放 `s` 倍，体积和质量会缩放 `s^3` 倍。归一化 mesh 若没有
真实尺寸、深度、标定相机或参照物，其真实质量不可辨识。

## 5. 概率分布的含义

系统输出 `mean/std/q05/q50/q95`。例如：

```text
q05 = 1.31 kg
q95 = 2.47 kg
```

表示模型约 90% 的后验质量样本落在 1.31-2.47 kg。它只有在独立测试集达到约 90% coverage
后，才能被称为校准良好的 90% 区间。区间宽不等于模型差，它可能诚实地反映内部不可见；
论文中必须同时报告区间覆盖率和宽度。

若希望得到类似 PUGS/NeRF2Physics 的单个值，可以使用后验均值或中位数，但不能丢弃区间。
最合理的论文呈现是“单点预测 + 不确定性区间 + 校准指标”。

## 6. VLM 微调

当前推荐起点是 **Qwen3-VL-8B-Instruct + QLoRA**。两张 RTX 4090 24 GB 可以进行 8B 级
多模态 QLoRA，建议每条样本 2-4 张视图、batch size 1/GPU、梯度累积、gradient
checkpointing 和 SDPA。

训练阶段：

1. 多视角 SFT：学习部件、材料和严格 JSON 输出。
2. 专业知识训练：加入有来源的材料和内部结构标注。
3. 质量弱监督：只使用训练集质量拟合校准器。
4. 心理物理 ranking/DPO：用于人类知觉一致性，不能替代物理真值。

当前代码实现了多图 LoRA/QLoRA SFT 入口；端到端可微质量 NLL 和 DPO 属于后续研究，不应
在当前论文中写成已完成。

## 7. 部件级 3D 编辑

编辑任务输入 source mesh、face labels、目标 part id 和 donor mesh，执行尺度匹配、对齐、
替换与物理重计算。检查项目包括：

- 接口 gap、穿透、自相交和连通组件数。
- 是否仍然水密，以及布尔融合是否成功。
- 替换前后质量、质心、惯量和支撑稳定性。
- 非编辑区域 Chamfer distance。
- donor 的语义与功能是否兼容。

当前 `part-replace` 是透明的 retrieval baseline，不是已经训练完成的生成式 Phy3D 编辑器。

## 8. 论文评测

对真实质量 `m` 和点预测 `p`：

```text
ADE  = mean(abs(p - m))                     越低越好，单位 kg
ALDE = mean(abs(log(p) - log(m)))           越低越好
APE  = mean(abs(p - m) / m)                 越低越好
MnRE = mean(min(p / m, m / p))              越高越好
```

这四项与 PUGS 发布代码的主要质量指标一致。项目还报告 RMSE、90% 区间 coverage、归一化区间
宽度、成对 bootstrap 95% 置信区间和方法间成对差异。

正式实验应包含：

- NeRF2Physics、PUGS、all-solid、solid/hollow、无 VLM、单视角和完整模型。
- 统一 ABO-500 测试对象，不能为某个方法单独删除失败样本。
- 至少三个随机种子；超参数只在 validation 选择。
- 单独报告 mesh 失败率、水密率和尺度误差。
- 部件密度、杨氏模量若无测量真值，只能展示定性结果。

## 9. 已实现代码

```text
src/tactile3d/abo500/                 ABO-500 数据与完整推理
src/tactile3d/mesh/                   修复、体积、渲染和部件替换
src/tactile3d/physics/interior.py     隐藏内部结构与质量积分
src/tactile3d/vision/                 多视角投影、OpenCLIP、VLM 客户端
src/tactile3d/training/               SFT、QLoRA 和心理物理拟合
src/tactile3d/eval/                   ADE/ALDE/APE/MnRE 与基线导入
src/tactile3d/figures/                论文定量图和部件物理图
```

主要入口：

```bash
bash scripts/run_abo500_mesh.sh DATA_ROOT MESH_ROOT 0.001 outputs/abo500_mesh
bash scripts/run_mesh_only.sh object.glb 0.001 outputs/object
bash scripts/train_vlm_2x4090.sh data/sft outputs/qwen3vl_8b_qlora
tactile3d mesh-volume --mesh object.glb --mesh-unit-to-meter 0.001
tactile3d paper-figures --run PUGS=pugs.json --run PartPhy3D=ours.json --output outputs/paper
tactile3d object-figures --prediction prediction.json --mesh mesh_metric.ply --face-labels face_labels.npy --output outputs/object_figures
```

## 10. 当前结论与下一步

论文第一主线应聚焦：**概率化隐藏内部结构的 mesh-native 物理 grounding**。部件编辑作为下游
验证或第二篇论文方向，避免第一篇论文同时承担过多主张。

执行顺序固定为：

1. 在 Linux 服务器准备 ABO-500 与对应 metric/watertight mesh。
2. 运行完整测试前，先用 `--limit 2` 验证数据、相机和 mesh 坐标一致。
3. 运行 all-solid、无 VLM、启发式内部结构和完整 VLM 四个消融。
4. 在训练集拟合质量校准器，锁定后运行 validation/test。
5. 导入 PUGS/NeRF2Physics 结果，生成统一指标、source data 和论文图。
6. 再收集专家内部结构标注，启动两张 4090 的 QLoRA 微调。

## 11. 相关文档

- `README.md`：安装和全部命令。
- `docs/research_design_zh.md`：完整研究问题、数据层级与实验边界。
- `docs/evaluation_protocol_zh.md`：指标、统计和论文报告协议。
- `docs/expert_annotation.example.jsonl`：专家标注格式。
- `docs/psychophysics.example.jsonl`：心理物理数据格式。
- `docs/references.bib`：基线论文 BibTeX。

本文件是 2026-09-15 讨论内容的固定快照。后续迭代请新建带日期的记录，并保留本版本。
