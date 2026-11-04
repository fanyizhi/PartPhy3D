# PartPhy3D 物理一致性编辑论文主线

> 本文件为早期探索记录。当前编辑主线、质量矩增量公式、局部更新条件与验证边界，以 [2026-10-30方法与贡献修订方案](ieee_trans_method_contributions_zh.md) 为准。下文建议的 `physics-edit` 尚不是已实现CLI命令；材料编辑不要求保持原总质量，力学约束也须有明确数据支持。

## 1. 中心问题

现有物理属性方法主要回答“这个物体是什么材料、可能多重”。本课题进一步研究：

> 当用户替换一个部件的材料或内部结构时，系统能否同步更新 occupancy 和刚体属性，自动
> 判断原几何是否仍能实现目标物理状态，并通过模拟验证后接受或继续修改？

这使工作从 physical property recognition 扩展为 **closed-loop physical editing**。

## 2. 统一部件物理状态

每个部件不再只有一个材料名称，而表示为：

```text
PartPhysicalState = {
  geometry,
  interfaces,
  surface_layers,
  interior_structure,
  occupancy_field,
  density_distribution,
  Youngs_modulus,
  Poisson_ratio,
  friction,
  uncertainty
}
```

`surface_layers` 区分皮革包覆层和内部泡沫/木框；`occupancy_field phi(x)` 表示位置 `x` 被
实体材料占据的比例。质量不是使用材料标签查一个常数，而是：

```text
m = integral rho(x) * phi(x) dx
c = integral x * rho(x) * phi(x) dx / m
I = integral rho(x) * phi(x) * (||r||^2 Identity - r r^T) dx
```

实现时可在水密 mesh 的填充体素或四面体网格上离散积分。总体积仍使用水密三角面的有向
四面体精确积分，体素只负责部件和内部材料分配。

## 3. 编辑类型

### 3.1 仅材料替换

几何和 occupancy 不变，只替换 `rho/E/nu/friction` 分布。例如木腿换成铝腿。系统必须同步
重算质量、质心、惯量和支撑载荷，不能只改变纹理或标签。

### 3.2 材料与内部结构同步

同时编辑表面层、壳厚、core/frame/infill/void 和 occupancy。例如：

```text
solid plastic seat
  -> leather surface + foam core + wooden frame + void
```

体积分数、填充率和空腔应满足守恒约束；VLM 输出候选分布，物理积分器负责归一化和拒绝
非法组合。

### 3.3 几何替换

当目标物理状态无法在当前几何内实现时，检索 donor part 或生成新几何，再检查接口、碰撞、
水密性和载荷能力。

## 4. Geometry adaptation gate

不要训练一个没有物理含义的“是否改几何”二分类器。先构造可解释约束：

```text
capacity violation: required occupied volume > available volume
shell violation: required wall thickness exceeds local geometric thickness
interface violation: estimated load / interface area exceeds allowable stress
stiffness mismatch: new and neighboring parts have incompatible compliance
stability violation: recomputed COM exits the support polygon
collision violation: edited geometry penetrates protected parts
uncertainty violation: probability of any violation exceeds threshold
```

根据约束分三级决策：

```text
level 0: no geometry adaptation
level 1: local shell, thickness, infill, or interface adaptation
level 2: full donor replacement or generative remeshing
```

第一版使用确定性规则和概率阈值，形成透明 baseline。后续 learned gate 以“最低编辑成本下
通过物理验证的最小 adaptation level”为监督标签。

## 5. 物理属性自动重构

编辑提交后，必须重新计算而不是在旧值上任意加减：

1. 重建每个部件的材料和 occupancy 场。
2. 对水密总体积重新积分。
3. 计算每个部件及整个物体的 mass、COM 和 inertia tensor。
4. 计算惯量主值和主轴，检查张量对称、半正定及三角不等式。
5. 更新接触面、摩擦、支撑多边形和接口载荷。
6. 将输入假设、近似方法和置信区间写入编辑报告。

当前 `part-replace` 已实现质量、质心、惯量张量、主惯量、接口 gap、一阶静态支撑和刚体
属性解析一致性检查；分层 occupancy 积分和编辑后全局重新体素化是下一版本需要补齐的核心
代码。

## 6. Simulation consistency

模拟不是最后放一张动画，而是编辑接受条件。采用从便宜到昂贵的分级验证：

### Stage A：解析预检查

- 质量、COM、惯量是否合法。
- COM 是否位于支撑多边形内以及稳定裕量。
- 部件是否碰撞、分离、自相交或非水密。
- 接口静载荷是否超过材料/连接允许值。

### Stage B：刚体模拟

对所有编辑运行固定协议：静置、倾斜、跌落和水平推力。记录：

```text
penetration depth
settling pose error
COM trajectory
linear/angular acceleration
energy drift
toppling and sliding outcome
```

模拟器必须使用本方法重新计算的 mass、COM、inertia 和 friction，而不是让模拟器根据均匀
mesh 自动生成另一套参数。

### Stage C：可变形模拟

只对泡沫、橡胶、织物等软部件运行有限元或其他连续体模拟，使用预测的 `E/nu` 和边界条件，
比较压缩量、接触面积和恢复过程。ABO-500 没有这些真值，因此需要合成有限元数据或真实
力-位移测量子集。

验证失败时返回具体违反项，由 adaptation gate 升级 level，而不是让 VLM 无约束地重新猜测。

## 7. 训练目标

```text
L = L_edit_semantics
  + lambda_geometry * L_protected_geometry
  + lambda_state * L_material_structure
  + lambda_mass * L_mass_COM_inertia
  + lambda_interface * L_interface
  + lambda_sim * L_simulation_consistency
  + lambda_cal * L_uncertainty_calibration
```

模拟奖励不能替代物理真值。优先使用可微约束或离线生成的验证标签训练，最后再用偏好优化
学习多候选编辑排序。

## 8. 实验与消融

按递增能力设计消融，正好对应论文故事：

```text
A  appearance-only geometry/material edit
B  A + material/property synchronization
C  B + interior structure and occupancy synchronization
D  C + automatic geometry adaptation gate
E  D + mass/COM/inertia reconstruction
F  E + simulation consistency loop (full model)
```

指标包括：

```text
Edit semantics: material/structure accuracy, instruction success
Geometry: protected-region Chamfer, interface gap, penetration, watertight rate
Physics: mass ADE/APE, COM distance, relative inertia-tensor error
Inertia: principal-moment error and principal-axis angular error
Gate: adaptation-level macro F1, over-edit rate, under-edit failure rate
Simulation: stable-settle rate, trajectory error, final-pose error, penetration
Uncertainty: 90% coverage and normalized interval width
```

ABO-500 只适合物体级质量验证。COM、惯量和编辑后模拟需要：

1. 使用 ABO mesh 构造材料/occupancy 已知的合成编辑对，得到精确 ground truth。
2. 使用带质量、COM 或力-位移测量的小规模真实物体子集验证 sim-to-real。
3. 将原始 ABO mesh 轨道与多视角重建 mesh 轨道分开报告。

## 9. 三个主要贡献

1. **Part physical scene graph**：联合表示表面材料、隐藏内部结构、occupancy 和不确定性。
2. **Minimum-cost geometry adaptation**：只在物理约束要求时修改几何，并选择最小修改等级。
3. **Closed-loop simulation verification**：编辑、刚体属性重构、模拟验证和失败反馈形成闭环。

论文主张应聚焦“编辑后的物理一致性”，而不是声称 VLM 能从外观确定唯一内部真值。

## 10. 建议代码结构

```text
src/tactile3d/editing/state.py          PartPhysicalState 与 EditSpecification
src/tactile3d/editing/synchronize.py    材料、内部结构和 occupancy 更新
src/tactile3d/editing/adaptation.py     adaptation gate 与约束违反概率
src/tactile3d/physics/mass_properties.py mass/COM/inertia 统一积分
src/tactile3d/simulation/validator.py   解析、刚体和可变形验证接口
src/tactile3d/editing/pipeline.py       候选生成、验证、升级和接受循环
```

建议新增统一入口：

```bash
tactile3d physics-edit \
  --source mesh.glb \
  --face-labels face_labels.npy \
  --prediction prediction.json \
  --edit edit_spec.json \
  --output outputs/edit_case
```

输出至少包括 `edited_mesh.glb`、`physical_state_before.json`、
`physical_state_after.json`、`adaptation_decision.json`、`mass_properties.json`、
`simulation_report.json` 和完整审计记录。
