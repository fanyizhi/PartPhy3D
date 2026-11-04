# PartPhy3D 科学研究设计

## 1. 研究目标

PartPhy3D 将两个相互依赖的任务放在同一套显式 mesh 表示中：

1. **物理 grounding**：从带真实尺度的纹理 mesh 推断部件、表面材料、不可见内部构造及物理属性分布。
2. **物理感知编辑**：替换、增加或修改一个部件，同时保持几何接口和物理可用性，并重算质量、质心和稳定性。

“触觉感知”在本文中必须谨慎使用。仅使用视觉和 mesh 时，系统预测的是**视觉条件下的触觉属性先验**。只有加入力-位移、GelSight、DIGIT 或材料实验测量后，才能声称融合了真实触觉观测。

## 2. 输入与输出

核心推理模块只要求：

```text
带纹理的三角网格 M=(V,F,T)
真实尺度 s，单位为 meter / mesh-unit
```

Blender 在程序内部生成固定相机轨迹的 RGB 图像。图像是 VLM 的内部观测，不是用户额外输入。无纹理 mesh 仍可运行几何分支，但材料后验应更宽，不能将形状猜测报告为视觉识别结果。

对部件 k，模型输出：

```text
p(material_k, structure_k, thickness_k, fill_k, density_k, E_k | mesh)
```

其中 `structure` 包括 solid、hollow、layered、foam-filled、frame-supported 和 partial-infill。一次后验样本的物体质量为：

```text
m = sum_k sum_j density[k,j] * occupied_volume[k,j]
```

代码通过 Monte Carlo 输出均值、标准差和 q05/q50/q95。`q05=1.31, q95=2.47` 表示约 90% 的模型后验质量落在 1.31–2.47 kg，而不是“真实质量必然有 90% 概率在其中”；只有在独立测试集上达到约 90% coverage 后，该区间才可称为校准区间。

## 3. 方法结构

```text
Textured metric mesh
    -> deterministic multi-view rendering
    -> semantic part segmentation
    -> multi-view surface material evidence
    -> VLM top-K latent interior hypotheses
    -> constrained material-volume allocation
    -> posterior mass / center of mass / modulus
    -> physics-aware part replacement
    -> geometric and static-physics verification
```

NeRF2Physics 和 PUGS 是对照方法，不应直接复制并改名。它们从多视角图像分别建立语言嵌入场和 Gaussian 表示。本项目保留它们的候选材料、特征传播、体积积分和评测思想，将几何载体改为显式、可编辑、可修复的 mesh。

## 4. 可检验的研究问题

### RQ1：显式 mesh 是否改善物理积分？

- H1：当尺度和几何质量受控时，水密 mesh 的体积积分比 NeRF/Gaussian 体积近似具有更低质量误差。
- 必须同时报告 mesh 重建误差；不能把更好的输入 mesh 带来的收益归因于物理模型。

### RQ2：分层内部模型是否优于实心/空心二分类？

- H2：对软包家具、容器和框架类物体，layered/frame/foam 模型降低 ALDE 和 APE。
- 对照：all-solid、binary solid/hollow、VLM scalar fill、完整结构图。

### RQ3：专业知识和心理物理先验是否互补？

- H3a：专业数据降低真实质量和材料属性误差。
- H3b：心理物理数据提高模型对人类重量、柔软度和稳定性判断的预测一致性。
- 两个目标必须分别评测。人类共识不是材料密度的物理真值。

### RQ4：物理约束是否改善部件编辑？

- H4：与只优化外观的替换相比，物理约束降低穿透、接口间隙和失稳率，同时保持非编辑区域。

## 5. 数据设计

### 5.1 物理真值层

- ABO-500：物体级质量与多视角图像，用于质量校准和主基准。
- 材料实验或可靠材料表：密度、杨氏模量、泊松比、摩擦和测试条件。
- 产品 BOM、CAD 或 CT：内部材料和构造的直接监督。
- PartNet/PartNet-Mobility：语义部件和 donor part 候选；不能假设其具有质量真值。

每条专业记录至少保存：`source_id`、测量/标准类型、单位、温度、方向、样品状态、许可和不确定度。随条件变化的材料数值应通过检索获得，不应只编码到模型权重中。

### 5.2 专家标注层

专家对每个部件给出：

- 表面材料概率；
- 2–4 个内部构造假设；
- 壁厚、占据率和填充率区间；
- 证据来源和置信度；
- “无法从外观确定”选项。

### 5.3 心理物理层

普通参与者完成受控的成对比较：

```text
只看渲染结果，A/B 哪个预期更重？
A/B 哪个看起来更柔软？
A/B 哪个更可能稳定？
```

优先采用成对比较而不是要求参与者填写精确 kg。`fit-psychophysics` 使用 Bradley–Terry 模型建立连续知觉尺度。实验需要预注册、知情同意、匿名化、注意力检查和独立被试测试集；具体伦理要求由所在机构审查。

### 5.4 防止泄漏

- 所有视角和部件按物理对象 `group_id` 一起切分。
- 同一商品、近重复 mesh 和同源 donor 不得跨训练/测试集。
- ABO 测试质量不能进入 prompt、RAG 或候选排序。
- 心理学验证集中的参与者和物体均不用于拟合。

## 6. VLM 微调

建议从 Qwen3-VL-8B-Instruct LoRA 开始。训练职责分为：

1. SFT 学习多视角、部件掩码和严格 JSON 输出。
2. 视觉适配学习 mesh 渲染域；数据不足时冻结视觉编码器。
3. 物理校准使用质量 NLL、体积约束和材料一致性损失。
4. 专家或参与者偏好足够后再增加 ranking/DPO，不以 DPO 代替物理真值。

基础目标可写为：

```text
L = L_sft + lambda_mat * L_material
          + lambda_mass * L_mass_nll
          + lambda_rank * L_human_pairwise
          + lambda_phy * L_volume_conservation
```

当前 `train-vlm` 实现标准多图 LoRA/QLoRA SFT。质量 NLL 和可微体积分支是下一阶段研究模型，不应把现有的训练入口描述为已完成该联合优化。

## 7. 部件级编辑

输入为 source mesh、face labels、目标 part id 和 donor mesh。可发表版本需要解决：

- donor 检索或生成；
- 语义和功能兼容性；
- 刚性/非刚性接口对齐；
- 无穿透、最小接口间隙和水密融合；
- 替换后的材料和内部结构；
- 质量、质心、惯量、支撑和接口载荷。

当前 `part-replace` 是明确标注的 retrieval baseline：均匀缩放、ICP、组件装配和一阶物理重算。它不会将未融合的装配伪称为制造级水密编辑；`edit_report.json` 会记录连通组件、体积方法和警告。

## 8. 评测与消融

### 物理 grounding

- 质量：ADE、ALDE、APE、MnRE、RMSE。
- 不确定性：NLL、90% coverage、区间宽度、ECE。
- 材料：面级 mIoU、macro F1、Brier score。
- 几何：watertight rate、体积误差、尺度误差。
- 内部结构：若有 CT/BOM，报告结构分类和材料体积误差；否则只做定性结果。

### 部件编辑

- 非编辑区域 Chamfer distance。
- 接口 gap、穿透率、自相交率和连通组件数。
- 水密率和布尔融合成功率。
- 质心位于支撑多边形内的比例与稳定裕度。
- 目标文本/部件语义一致性。

### 必需消融

```text
all-solid
binary solid/hollow
no VLM
single view
no professional evidence
no psychophysical prior
no mass calibration
no physics editing constraint
full model
```

至少运行三个随机种子，报告均值、标准差和置信区间。模型选择只看 validation，test 只运行最终配置。

## 9. 当前代码成熟度

| 模块 | 状态 | 科学边界 |
|---|---|---|
| ABO-500 读取与质量评测 | 已实现 | 需要用户提供合法数据和 metric mesh |
| 水密检查、分割、体积积分 | 已实现 baseline | 几何聚类不是语义分割真值 |
| VLM 内部结构 JSON | 已实现接口 | 未调用用户服务器完成真实训练 |
| 概率质量/模量输出 | 已实现 | 模量没有 ABO 部件级真值 |
| mesh-only 自动渲染 | 已实现 Blender 入口 | 运行机器需要 Blender |
| 专家 SFT 数据准备 | 已实现 | 需要真实专家标注 |
| 心理物理尺度拟合 | 已实现 | 需要伦理合规的人类数据 |
| Qwen3-VL LoRA/QLoRA | 已实现训练入口 | 参数需按服务器显存验证 |
| donor 部件替换 | 已实现 baseline | 尚非可微生成式编辑器 |
| 物理约束生成式编辑 | 研究阶段 | 需 PartNet 配对与仿真监督 |

## 10. 推荐论文结构

主论文聚焦一个中心贡献：**概率化隐藏内部结构的 mesh-native 物理 grounding**。部件编辑作为该表示的重要下游任务。若后续编辑数据和结果足够强，再拆分为第二篇论文，避免第一篇同时承担过多未经充分验证的主张。
