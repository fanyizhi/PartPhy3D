# 编辑指令执行器：运行、验证与边界

本次交付是现有共享部件物理状态引擎的指令入口，支持明确中文句式和严格
JSON程序。默认使用已知构造装配体验证材料、封闭盒壳和安装兼容替换件。
没有调用千问、InstantMesh或PartField，没有启动服务器任务。

实际执行记录（2026-11-04）：39项相关测试通过；最终绘图调整后，包内替换件冻结、
三类图片导出和重放测试再次通过。六个变更Python文件通过88列E/F/I静态检查；
Python 3.10语法检查通过，实际运行是本机Python 3.13。

最终演示在`outputs/edit_commands_20261104_v2/`；独立命令行重放在
`outputs/edit_commands_replay_20261104/`。四个状态的质量完全相同。
默认合成解析参考的最大绝对误差：质量1.11e-16 kg、质心1.11e-16 m、
惯量2.53e-16 kg·m²，属于浮点舍入量级；这不是实际物体预测精度。
三个步骤的局部更新与全量重算原始质量矩之差均为0。

## 直接运行

在项目根目录、已经安装项目依赖的Python环境中：

```bash
python scripts/edit_parts.py
```

输出到新的`outputs/edit_commands/run_<时间戳>/`，不会覆盖已有实验。
源文件入口是`src/tactile3d/experiments/edit_commands.py`，入口脚本自动添加
项目`src`目录，不依赖之前遇到问题的全局`tactile3d`命令。

默认四个状态：

| 状态 | 操作 | 解析总质量/kg |
| --- | --- | ---: |
| S0 | 已知木质主体与实心木腿 | 0.7840 |
| S1 | 木腿材料改成示例钢材 | 3.0880 |
| S2 | 钢腿改为10mm壁厚的封闭空心盒 | 2.5228 |
| S3 | 安装兼容的台阶形铝腿替换原腿 | 0.8568 |

密度是指定的合成参数，不能解释为ABO物体的识别结果。解析参考通过长方体
闭式惯量和移轴定理计算，包括空腔扣除、替换件分段以及整体质心移动。

## 支持的中文句式

```text
将 leg 的材料 wood 改为 steel，密度 7850 kg/m3
将 leg 设为空心盒，材料 steel，壁厚 10 mm
将 leg 设为实心，材料 steel
将 leg 替换为 aluminum_leg，接口 mount
```

允许`把`代替`将`、中英文逗号、末尾句号、mm/cm/m及毫米/厘米/米。
不会猜测旧材料、密度、壁厚、安装接口或视角左右。不支持的附加文字整句拒绝，
不会只执行成功匹配的前半句。每条`--command`只包含一个操作。

对内置合成体执行两步：

```bash
python scripts/edit_parts.py \
  --command '将 leg 的材料 wood 改为 steel，密度 7850 kg/m3' \
  --command '将 leg 设为空心盒，材料 steel，壁厚 10 mm'
```

对已有共享物理状态执行编辑：

```bash
python scripts/edit_parts.py \
  --state /absolute/path/to/state_directory \
  --alias '左前腿=leg' \
  --command '将 左前腿 的材料 wood 改为 steel，密度 7850 kg/m3'
```

替换操作还需提供`--donor 'aluminum_leg=/absolute/path/to/donor_state'`。
源状态目录和替换件目录都包含`state.json`及`cells.npz`；替换件恰含一个部件。
不是任意GLB/OBJ即插即用，需要先建立共同网格、材料占用和安装接口。

## JSON程序与重放

完整程序根字段固定，未知字段会被拒绝：

```json
{
  "schema_version": 1,
  "object_id": "synthetic_mount_assembly",
  "expected_object_version": 0,
  "steps": [
    {
      "target_part_id": "leg",
      "operation": "material_change",
      "expected_part_version": 0,
      "material_name": "wood",
      "new_material": {
        "name": "steel",
        "density_kg_m3": 7850.0,
        "source": "explicit_design_input"
      }
    },
    {
      "target_part_id": "leg",
      "operation": "construction_change",
      "expected_part_version": 1,
      "material_name": "steel",
      "construction": "hollow_box",
      "wall_thickness_m": 0.01
    }
  ]
}
```

物体版本约束整组指令，部件版本可进一步限制单步。未指定部件版本时，执行器
从同一事务的当前状态绑定版本，再将明确版本写入输出程序。替换路径相对JSON
文件所在目录解释；导出程序将实际使用的替换件状态复制到包内并改用相对路径，
可以把整个结果目录传到Linux后重放。

```bash
python scripts/edit_parts.py \
  --state /absolute/path/to/result/state_000 \
  --plan /absolute/path/to/result/program.json
```

重放结果不受原替换件目录后续修改影响。第一次执行已经在内存中固定实际使用
的替换件；导出复制该快照，而不重新读取可能变化的外部文件。

## 输出与检查

- `program.json`：带实际部件版本、包内替换件路径的可重放操作。
- `state_000/`至最终状态：JSON/NPZ物理状态和编号PLY边界，编号与部件对应在报告中。
- `donors/`：实际使用的替换件快照。
- `physical_updates.csv`：各状态质量、质心、主惯量；默认案例附解析质量与误差。
- `report.json`：逐步更新/复用部件、接口检查、全量核对、解析参考与限制。
- `manifest.json`：输出文件SHA256、算法源文件SHA256及Python/NumPy版本。
- `physical_updates.{svg,pdf,png}`：x为状态编号；y分别为质量kg、质心m、惯量kg·m²、部件计数。
- `geometry_updates.{svg,pdf,png}`：世界坐标中的体素边界及质心，未渲染原纹理。
- `density_slices.{svg,pdf,png}`：第一个编辑目标在各状态中部y层的xz有效密度截面，单位kg/m³。

图中显示逐状态原始计算，没有平滑、置信区间或统计显著性。超过四个状态时，
几何与截面图等距选取四个状态，编号记录在报告中，所有数值状态仍完整导出。
截面使用各状态最接近部件中部的网格y层；新旧部件姿态不同不代表同一材料切面。

所有步骤在内存中完成并验证后才创建结果目录。计算失败时原状态保持不变，
整组不会保存部分成功结果。文件写入失败可能留下未完成目录；只有完整写出
`manifest.json`且哈希匹配才视为完整导出，重试请使用新目录。

逐步时间分别记录编辑（含约束检查）和完整核对，不是相同工作范围的严格性能
基准，不能据此直接声称加速倍数。缓存计数也不代表端到端推断速度。

## 参数与局限

- Python语法要求3.10及以上；本机实际执行环境3.13，Linux运行尚未验证。
- `--state`、`--plan`、`--command`、`--alias`、`--donor`、`--output`详见`--help`。
- `physics/edit_program.py`：最多64步，每句最多1024字符，单位换算与句式集中定义。
- `figures/edit_history.py`：预览180dpi、最多四个几何面板、截面最多200万个像素。
- `experiments/edit_commands.py`：解析参考绝对/相对容差均为1e-10。
- 当前材料编辑同步有效密度、质量、质心及惯量；尚未同步杨氏模量、摩擦或联合不确定性。
- 几何域要求共同轴对齐体素网格；任意角度姿态需要尚未实现的重体素化。
- 空心编辑仅支持完整矩形域的封闭盒壳，壁厚必须为整数体素层且存在可分辨空腔。
- 当前示例10mm体素不能表达2mm壁厚，程序明确拒绝，不会替换成包围盒或偷偷取整。
- 安装检查是接口位置、方向、尺寸和接触采样，不能证明真实连接强度。
- 中文解析是确定性受限语法，不是已经微调好的千问；未来VLM输出也须经过同一JSON校验。
- 原始物体及编辑后真实物理精度、物理场学习、跨LOD映射、机器人验证仍需后续实验。

新增运行依赖为零；沿用NumPy、SciPy、Trimesh和Matplotlib，测试使用pytest，格式检查使用ruff。
