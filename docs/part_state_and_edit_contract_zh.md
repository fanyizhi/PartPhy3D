# 统一部件物理状态与编辑：数据契约和实现边界

更新：2026-11-03。本文解释统一表示和部件编辑的目标契约。轴对齐共同体素域的确定性首版已实现并运行，详见 [实现与验证记录](part_state_implementation_2026-11-03_zh.md)；学习式融合和完整目标契约尚未完成，也未证明原创性。

## 1. 同一份物理状态的三个层次

数据组织为Object → Part → ConstructionRegion/Layer → VolumeCell。表面Mesh是这一状态的几何载体，关系图另行记录部件间固定/接触/未知连接，不把关系图硬塞成唯一父子树。

1. **对象层**：对象ID、SI单位、共同世界坐标系、部件集合、关系图、状态版本、输入与模型来源。
2. **部件层**：稳定安装槽位`part_id`、具体实例`instance_id`、局部到世界变换、表面/体积支持、接口、材料与构造候选、融合特征、质量矩与依赖版本。
3. **局部场层**：可追溯的表面锚点、非重叠体积单元、几何域、各材料占用率、材料参数及其联合不确定性。

同槽位换donor可以保留`part_id`，但应产生新的`instance_id`和版本；只修改原实例材料则更新版本。层次拆分/合并要记录显式谱系，不能仅因数字标签相同就认为部件对应保持成立。

拟议`PartPhysicalState`字段：

```text
identity: part_id, instance_id, parent_id, version
geometry: mesh_ref, face_support, local_to_world, evidence_anchors
features: visual, geometric, semantic, fused_token
construction: hypotheses, regions, thickness, occupancy_field_ref
material: class_beliefs, parameter_distributions, provenance
interfaces: anchors, normals, patch_shape, dimensions, relation_type
physics: effective_density_field_ref, raw_mass_moments, uncertainty_samples
dependencies: source_versions, operator_versions, invalidation_records
```

字段相同只是工程统一，不等于学习对齐或算法创新。原始数值、单位和证据来源保留在显式字段中，不依赖一个不可解释Token承担全部物理量。

## 2. 空间统一与学习式映射分别实现

**空间统一**：图像像素经相机/可见性投影关联Mesh表面锚点，锚点关联稳定部件和部件局部坐标，再关联内部体积支持。图像裁剪同步内参。第一版可保存表面三角形及重心坐标、世界位置和局部位置，减面后重新建立候选对应并检验误差；开放表面不直接获得体积。

最近三角形中心、最近体素或普通ICP只是近似基线，可能跨越邻近薄层、材料界面和接口。目标映射要结合部件身份、可见性、距离/法向及接口约束，在无法建立对应时显式拒绝或重新推断。旧三角形ID在减面后无效，不能直接沿用。

**学习式统一**：视觉、几何、材料语义特征分别经投影进入同维隐空间，再用融合模块生成部件Token。预测头结合Token及显式局部几何，输出构造候选与物理字段。使用训练集对应、材料/构造、质量及可用编辑配对监督；同维投影与拼接本身不保证语义对齐。

只有总质量时，密度与占用仍可能相互补偿。需要独立构造/局部标签或已知构造的合成数据来检验具体机制；VLM伪标签与实测标签分开，避免以模型输出验证自身。

## 3. 材料类别概率与空间占用不同

材料类别置信度表示某处可能是什么材料；alpha_k(x)表示一个确定/采样构造中的实际材料体积分数，两者分开保存。含空腔的各组分占用非负且总和为1。

在一个构造样本内，rho_eff(x)=sum_k alpha_k(x)*rho_k(x)，以有限体积单元积分质量、COM和完整惯量。泡沫密度已包含微孔时不再按相同孔隙重复折减。部分覆盖单元应记录真实支持/近似阶次，不能把所有边界单元默认当完整实心立方体并称精确。

体积单元不重复归属部件，或使用明确的组装规则消除重叠；表面涂层与内部层不要重复计量。未知E、摩擦等保留未知或标为先验，不能由质量倒推为真值。

## 4. 第一版编辑操作与输入

实现三种可解释操作：`material_change`、`construction_change`、`replace_part`。具体编辑要求由`EditSpec`声明。换钢不自动推出空心；壁厚、空腔和尺寸需要用户指定、构造候选或明确设计约束。

以下JSON仅为目标格式示意，donor路径与部件ID是示例：

```json
{
  "target_part_id": "chair/leg_1",
  "operation": "replace_part",
  "donor_ref": "donors/steel_leg.glb",
  "construction": {
    "type": "hollow_shell",
    "material": "steel",
    "thickness_m": 0.002
  },
  "constraints": {
    "preserve_mount_interface": true,
    "preserve_other_part_geometry": true
  }
}
```

上例2mm只是示例编辑参数，不是自动设计出来的安全壁厚。接口位置/方向/尺寸与间隙等容差应另在配置中给出、说明来源；没有连接结构、载荷和强度数据时不判断真实承载能力。

## 5. 编辑事务的六个步骤

1. **校验与快照**：检查目标身份/版本、参数单位、材料和donor；保存源状态，失败可撤销。
2. **接口对齐**：利用安装锚点、方向和接口尺寸求允许的变换，检查穿插/间隙及保护区域。第一版优先接口兼容donor；不以整体包围盒缩放或ICP代替装配可行性。
3. **同步构造**：更新表面和内部层、壁厚/占用、材料分布及部件局部支持；几何变化的体积域重新离散化，避免把旧体素坐标直接套到新donor上。
4. **传播依赖**：目标字段失效后，按算子依赖更新受影响接口及关联部件；没有改变输入或依赖的缓存才可复用。全局几何/载荷/概率关联不能只按空间距离截断。
5. **重算并验证**：在共同世界坐标下更新质量矩，检查总质量、COM、完整惯量、占用有效性、接口和保护区；与同条件全量重算比较。
6. **提交或拒绝**：通过已声明约束后生成新版本、编辑资产和报告，否则保留源版本并给出违反项。几何兼容不等于真实连接强度认证。

材料参数变化未必需要重新体素化；体积支持和分层占用发生几何变化时才更新相应域。每次编辑不重复运行InstantMesh；需要新形状时输入明确donor或单独生成候选。

## 6. 局部质量矩更新条件

在空间支持不重叠、坐标一致、受影响集合A完整且其余贡献确实不变时，缓存每部件的原始质量矩(m_i, s_i, Q_i)，其中s_i=integral(x*rho_eff dx)，Q_i=integral(x*x^T*rho_eff dx)。有限单元内部二阶矩也要计入。

```text
(m, s, Q)_new = (m, s, Q)_old
              - sum_{i in A}(m_i, s_i, Q_i)_old
              + sum_{i in A}(m_i, s_i, Q_i)_new
COM = s / m
I_COM = trace(Q)*Identity - Q
        - m*((COM dot COM)*Identity - COM*COM^T)
```

不能直接相加各部件在自身质心处的惯量。材料/结构编辑允许改变总质量，不要求编辑前后质量守恒。共享材料密度、全局尺度等随机变量造成的相关性需保留；存在关联时不能独立刷新单部件随机样本并宣称正确更新不确定性。

质量矩加性是已有数学性质，不单列为新定律。新增机制需由共享表示、联合构造更新、接口条件及依赖识别的可测效果支持。

## 7. 最小验证与当前代码差距

先用已知几何和材料的实心/空心/复合部件，验证材料编辑、构造编辑、donor替换及坐标变换。比较增量与全量结果；接口失败、标签映射失败、负占用和体积重叠应被明确拒绝。再评价实际物体的独立质量/编辑真值。

已有代码：`physics/unified.py`提供不可变规范立方单元与每部件常密度的积分、近邻显示查询；`physics/fields.py`提供逐单元密度积分；`mesh/editing.py`提供均匀缩放/ICP donor拼接、有效密度和刚体质量量重建。

首版新增：`physics/part_state.py`、`part_editing.py`及`part_io.py`已实现显式部件状态、严格EditSpec、确定性缓存更新、矩形接口及材料支持采样、材料/盒壳编辑和兼容体素donor装配。`mesh/state_surface.py`提取体素材料边界，不使用凸包填空腔。16项新增测试及5项既有相关回归通过，另完成10例缓存字段迁移审计。

尚未完成：任意姿态重体素化、自由分层构造、可靠跨LOD锚点迁移、学习式跨模态对齐和全局推断/力学依赖传播。旧`mesh/editing.py`的凸包/ICP基线单独保留，不与新体素引擎混称。
