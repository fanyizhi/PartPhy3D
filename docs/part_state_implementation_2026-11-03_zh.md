# 部件统一状态与编辑首版：实现、测试及复现

日期：2026-11-03。已在本机Python 3.13上执行，不使用GPU或远程服务器。本文是软件与数值验证记录，不是学习模型、真实物理准确性或论文创新性结论。

## 1. 本次完成

- `physics/part_state.py`：不可变部件身份、版本、SI坐标、材料/空腔占用、密度字段、安装接口、连接图和共同体素域。重复/重叠体素、非刚体变换、错误占用和不兼容离散条件明确拒绝。
- `physics/part_editing.py`：严格`EditSpec` JSON读写，材料修改、实心/封闭盒壳构造修改、接口兼容的已体素化donor替换；失败不修改旧状态。
- `physics/part_io.py`：JSON与NPZ状态存储、NPZ哈希核验、源输入不覆盖；导入旧10物体有效密度均值时不重复乘占用率。
- `mesh/state_surface.py`：提取正密度单元的显式边界，不用凸包填内部空腔。仅代表体素材料域，不是原图纹理化Mesh。
- `experiments/part_edit_demo.py`及`scripts/demo_part_edit.py`：解析示例、JSON编辑重放、10物体旧字段迁移检查、Matplotlib图与PLY导出。

统一状态包括Object → Part → MaterialPhase/Cell。`feature_refs`只记录引用，尚未训练或实现跨模态特征对齐；不把这一工程表示称作已完成的学习式统一Token。

固定安装接口检查位置、完整安装方向、尺寸，并在矩形接口两侧按当前体素尺度采样实际材料支持。它不是精确CAD接触求交或连接强度证明。

## 2. 数值证据

解析例子为矩形主体和一个支撑部件，参数与构造完全已知，不是ABO预测物体，也不用于宣称稳定支撑或机械臂成功。

| 状态 | 整体质量kg |
| --- | ---: |
| 原始木质支撑 | 0.7840 |
| 同几何替换为实心钢 | 3.0880 |
| 钢支撑改为封闭空心盒壳 | 2.5228 |
| 替换为安装面兼容的阶梯铝支撑 | 0.8568 |

盒壳示例体素和壁厚均为1cm，服务于解析验证，不是实际产品安全设计。材料变化、构造变化和几何替换分别验证，未自动从“钢”推断空心或壁厚。

新测试加已有相关回归共21项通过，覆盖：解析质量/完整惯量、旋转和平移、闭合盒壳体积与惯量、材料占用保持、槽位和实例身份、缓存复用、版本冲突、接口尺寸及材料支持、体素重叠、任意角度拒绝、亚体素壁厚拒绝、部件内绑定、序列化/篡改检测、JSON字段/类型和避免重复折减占用。

示例每次只重新积分目标部件，未变部件复用原质量矩；与完整重算的m、一次矩和二次矩误差均为0（此次浮点实现下）。对象级重叠和接口检查仍执行，不能据此声称端到端计算只访问一个部件或已实现某个加速倍数。

已有10物体均成功转换为新状态、保存/加载并执行显式有效密度字段×1.1的数值情景。每次重算1个区域，其余7个区域质量矩复用。导入前后总质量最大差约7.11e-15 kg，局部更新与全量二次矩重算差为0。

该10物体检查不读取真实质量，也没有重新预测材料或真实编辑结果。旧几何聚类不是语义部件，旧不确定性样本及相关性没有迁移，旧预测准确性没有因此提高。

## 3. 本次结果位置

- `outputs/part_state_edit_demo_v2/report.json`：解析例子和三次编辑。
- `outputs/part_state_edit_demo_v2/state_0`至`state_3`：JSON/NPZ状态及逐部件PLY。
- `outputs/part_state_edit_demo_v2/donor`：单部件替换件的可加载状态。
- `outputs/part_state_edit_demo_v2/edit_specs`：可重放的三类编辑JSON。
- `outputs/part_state_edit_demo_v2/01_geometry_edits.{png,svg,pdf}`：几何和COM展示。
- `outputs/part_state_edit_demo_v2/02_physical_state.{png,svg,pdf}`：质量、COM、主惯量与空腔切片。
- `outputs/part_state_ten_audit_v1/audit.csv`：旧10物体字段的数值迁移/编辑检查。
- `outputs/part_state_json_edit_v1`、`part_state_json_shell_v1`、`part_state_json_replace_v1`：单独CLI重放结果。

## 4. 最短复现

在项目根目录，用已有依赖环境执行（无需服务器）：

```bash
python scripts/demo_part_edit.py
```

默认输出到`outputs/part_state_edit_demo/run_<编号>`，重复运行创建新目录。也可指定新输出目录，已有目录会拒绝覆盖：

```bash
python scripts/demo_part_edit.py --output outputs/my_part_edit_run
```

重放一次材料编辑（状态目录已由上一步创建）：

```bash
python scripts/demo_part_edit.py \
  --state outputs/my_part_edit_run/state_0 \
  --edit-spec configs/edits/wood_to_steel.json \
  --output outputs/my_material_edit
```

构造编辑以`state_1`为输入，使用该次示例目录`edit_specs/2_construction_change.json`；替换以`state_2`为输入，使用`edit_specs/3_replace_part.json`。替换件由编辑JSON中的相对目录定位，无需传私钥、网络或模型令牌。

对现有10物体字段进行审计：

```bash
python scripts/demo_part_edit.py \
  --pilot outputs/abo500_ten_v2 \
  --output outputs/my_ten_state_audit
```

依赖：NumPy、SciPy、Trimesh；示例出图另需Matplotlib。测试需pytest，代码检查用ruff。未新增第三方包。

在Windows的本机环境，`python`未配置时使用`py -3.13`替换即可。

测试命令（项目已可导入或设置`PYTHONPATH=src`）：

```bash
python -m pytest -q -p no:cacheprovider \
  --basetemp outputs/pytest_part_state_local \
  tests/test_part_state_editing.py tests/test_unified.py \
  tests/test_editing.py tests/test_physics_lod.py
```

本机默认临时目录存在权限问题，测试指定项目内新临时目录后通过；未删除或修改其他用户的临时目录。

## 5. 首版的明确限制和下一步

- 共用轴对齐、同边长体素网格；允许保持网格的90度旋转与对齐平移，任意角度donor需要重体素化，当前明确拒绝。
- 构造编辑仅支持固定域实心化和完整矩形域的整数体素厚度闭合盒壳。自由薄壳、分层泡沫/织物和复杂内部骨架未完成。
- donor必须是明确坐标、材料和接口的体素状态，不能把任意GLB直接冒充可编辑物理部件。
- 单元内部按均匀相密度积分；占用混合是该离散模型内的近似，未恢复真实亚体素结构。世界/对象坐标应靠近对象，避免大坐标下原始二阶矩相减失去精度。
- 基础表面绑定限定部件身份与距离，未实现跨LOD重心锚点迁移、跨视角学习对应、自动语义部件识别或纹理重绘。
- 依赖复用只针对本引擎显式确定字段；全局学习推断、跨部件概率相关、应力/形变和承载验证不包含在内。
- InstantMesh、Qwen、PartField真实模型链及任务约束稀疏化仍按当前流程单独推进。本次未执行服务器训练或任何队列操作。

下一阶段是将真实分割得到的部件/接口与上述状态连接，补充几何重体素化及可核验构造监督，再实现与消融学习式融合；不将当前数值正确性升级为算法原创性或预测优势。
