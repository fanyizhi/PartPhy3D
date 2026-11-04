# ABO-500单对象物理预测实验记录

日期：2026-10-30。对象：`B07B4G5RBN_ATVPDKIKX0DER`，带木框和软包的贵妃椅。

本次已运行代码并输出数值、字段、图表与审计记录。它是未训练的构造先验基线和几何诊断，不代表拟议论文系统已经完成，也不是T1期刊水平的性能结论。

## 1. 数据与选择依据

- 数据位置：`D:/06-数据集/abo_500`。
- 选择规则：发布版`splits.json`中test列表的第一个ID；该文件有300/100/100个训练/验证/测试对象。
- 样本有30张带alpha轮廓的1024×1024图像和标定相机。
- 原始Mesh已缓存到`data/abo500_single/B07B4G5RBN_ATVPDKIKX0DER/model.glb`，9490个三角形，5623个顶点，非水密。
- 模型来源：[ABO官方原始资产](https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/N/B07B4G5RBN.glb)。数据与派生标签按ABO的CC BY 4.0要求注明Amazon.com及ABO作者归属。[数据说明](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)。
- 渲染坐标固定采用`(x,y,z) -> (x,-z,y)`，单位比例1；没有根据质量拟合尺度。
- Mesh与30个数据集视角的平均轮廓IoU为0.982768，验证几何对应；这不是部件分割或物理属性准确性。

此前下载的`B075X4QMX3`沙发不在当前ABO-500划分中，故本次没有继续用它冒充ABO-500样本。

本对象已用于检查与开发几何规则。虽然其来源属于官方test划分，也不能再将它作为本项目未接触过的最终盲测样本。后续算法选择应使用train/val，并将本开发案例在最终泛化评价中单列或明确排除。

## 2. 本次实际运行的方法

1. **原物体保持不变**：没有执行材料替换、部件替换或结构编辑。
2. **几何部件候选**：按原生连通片、位置、大小和高度规则划分11个区域：木框、4条腿、软包底座、座面、靠背、活动靠垫、左右扶手。原生连通片约43个，不等同于制造部件真值；这不是SAM、PartField或Qwen的预测。
3. **多视角轮廓约束**：在米制坐标中对规则网格做30视角轮廓交集，得到可见证据支持的外包络。轮廓直接使用数据集alpha，是额外输入；不能与只输入RGB的模型混为相同输入预算。
4. **几何约束裁剪**：暴露的木部件限制在原生组件包围盒内，删除不受木构件支持的椅下空间，防止轮廓包络把高密度木材扩展到空处。这是针对当前对象的几何基线；包围盒不是任意复杂部件的精确体积。
5. **分层局部物理字段**：固定织物/填絮层厚度，内部用木框、泡沫和宏观空腔的条件先验表达。材料类别与体积占用分开，各组分含空腔总和为1；每个体积单元唯一归属，避免重叠双重计质量。
6. **共享先验采样**：4096次采样，种子20261030。材料密度在不同部件间共享采样，保留相关性。泡沫密度为表观密度，额外空腔仅描述宏观空间。层厚、几何和规则标签的不确定性没有全部纳入区间。
7. **积分**：从空间密度积分质量、质心和完整惯量张量，包含有限立方体单元自身的惯量。
8. **保存再评测**：先保存配置、字段、预测、SHA-256校验记录，再由独立evaluate阶段读取产品质量标签。

密度范围来自本对象发布版`info.json`中的缓存材料建议：木材300–700、泡沫20–60、织物100–300、填絮10–50 kg/m³。内部比例为明确写入配置的人工假设，没有经过训练或概率校准。没有声称这些缓存建议是本次新调用Qwen得到的。

## 3. 主结果：只代表这一件物体

数据集质量标签：**29.029888 kg**，来自产品元数据；本次没有用电子秤独立复测实物。

主分辨率12 mm，体积单元275892个，几何包络体积0.476741376 m³。

| 同几何下的构造假设 | 质量kg | ADE kg↓ | ALDE↓ | APE %↓ | MnRE↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全物体按实心织物 | 94.944247 | 65.914359 | 1.184964 | 227.056884 | 0.305757 |
| 木构件＋实心泡沫 | 23.288124 | 5.741764 | 0.220382 | 19.778803 | 0.802212 |
| 分层构造先验 | **29.114855** | **0.084967** | **0.002923** | **0.292688** | **0.997082** |

分层模型90%先验区间为**21.552003–36.529323 kg**，不是经过验证集校准的置信承诺。单个标签落入该区间不能说明覆盖率达到90%。

质心为`[0.003519, 0.368484, -0.169791] m`，位于原始glTF资产坐标系；完整惯量保存在`prediction.json`。两者没有独立实测参考，因此只报告模型输出，不报告“真实质心/惯量误差”。

本例非常接近质量标签，仍可能存在几何、构造和密度误差抵消。没有部件密度/内部结构真值，不能据此证明内部恢复正确，更不能据此宣称优于论文基线。

## 4. 几何诊断与保留的早期结果

在首次读取本对象质量标签之前，按几何诊断先后运行了三个版本；材料密度及内部占用先验保持不变。以下完整保留，不隐藏早期失败：

| 几何处理 | 主分辨率质量kg | APE % | 结论范围 |
| --- | ---: | ---: | --- |
| 开放Mesh逐部件体素闭合 | 20.781458 | 28.413579 | 对分辨率敏感，不能当作精确体积 |
| 仅多视角轮廓包络 | 53.218368 | 83.322677 | 未见凹陷与空处可能被填入；木材归属扩散 |
| 轮廓＋几何支持约束 | 29.114855 | 0.292688 | 当前对象上的诊断改进，尚非泛化证明 |

保存位置分别为`outputs/abo500_single_object`、`outputs/abo500_single_object_v1`和`outputs/abo500_single_object_v2`。第三版为本次主结果。

主方案的分辨率敏感性：

| 网格边长mm | 包络体积m³ | 质量kg |
| ---: | ---: | ---: |
| 20 | 0.468168 | 28.788463 |
| 12 | 0.476741 | 29.114855 |
| 8 | 0.479489 | 29.043891 |

12→8 mm的质量变化约0.244%，体积变化约0.576%。这是离散稳定性，不是相对真实内部体积的误差；主结果固定为预设12 mm，没有按接近标签程度选择分辨率。

## 5. 物理属性图如何解释

`02_physical_fields`包含有效密度、占用率、木材占用、密度先验变异系数、理想化弹性上界与内部剖面。

- 密度与占用来自假设字段，未获得逐点真值。
- 杨氏模量图采用明确假设的各向同性相常数，经体积/剪切模量的Voigt/Reuss混合界计算。它不是测得的有效模量，也不是夹爪接触刚度。含空腔区域的Reuss下界可能为0，表示信息不足，不能仅展示上界并称其为真实刚度。
- 木材各向异性、织物、泡沫非线性均未被该简化模型准确覆盖。
- 没有指定夹爪接触材料对和独立测量，因此未伪造摩擦、硬度准确性。
- 表面可视化在同一部件内进行4近邻插值；积分使用原体积字段，没有使用显示插值修改质量。

已补充`src/tactile3d/eval/properties.py`及待测参考CSV。当前局部属性没有独立参考，评分JSON明确为`reference_unavailable`；接口、单位和正式对比协议见`docs/local_property_evaluation_zh.md`。新增接口没有修改本次已保存的质量或物理字段。

## 6. 已交付文件

主输出：`outputs/abo500_single_object_v2`。

| 文件 | 内容 |
| --- | --- |
| `01_multiview_parts.png/svg/pdf` | 两个视角的RGB、匹配Mesh和部件候选 |
| `02_physical_fields.png/svg/pdf` | 物理假设字段和内部剖面 |
| `03_mass_evaluation.png/svg/pdf` | 质量对比、先验分布、部件质量和四项指标 |
| `04_numerical_audit.png/svg/pdf` | 分辨率敏感性与30视角对应检查 |
| `mass_metrics.csv` | 可追溯的逐假设质量指标 |
| `geometry_ablation.csv` | 三种几何处理的单例数值对照 |
| `part_physical_parameters.csv` | 11个部件的体积、密度、占用与质量区间 |
| `physical_field.npz` | 米制体素中心、归属、材料占用、密度、弹性界与质量样本 |
| `part_labels.ply`, `face_labels.npy` | 带标签Mesh及原面对应 |
| `prediction.json`, `evaluation.json` | 预测与标签评价分离的记录 |
| `frozen_experiment_spec.json`, `prediction_seal.json` | 输入、配置、预测阶段代码及产物校验 |

预测阶段代码校验保留当时版本；后续仅修改过绘图裁剪和表面显示插值。预测与字段校验保持不变。当前源码可重现相同物理计算，但完整文件哈希因显示代码更新而变化。

## 7. 复现

代码入口：`src/tactile3d/experiments/single_object.py`。配置：`configs/single_object_chaise.yaml`。主要计算只需NumPy、SciPy、Trimesh、Pillow、PyYAML和Matplotlib，不需要GPU。

Windows当前电脑可直接运行，使用新输出目录保留本次记录：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
py -3.13 -m tactile3d.experiments.single_object --root 'D:\06-数据集\abo_500' --mesh data/abo500_single/B07B4G5RBN_ATVPDKIKX0DER/model.glb --config configs/single_object_chaise.yaml --output outputs/abo500_single_reproduced --stage all
```

Linux/MobaXterm在已安装本项目的环境、项目根目录中执行；把ABO_ROOT改成服务器已有数据集目录：

```bash
export ABO_ROOT=/服务器上的实际路径/abo_500
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p data/abo500_single/B07B4G5RBN_ATVPDKIKX0DER
curl --fail --location 'https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/N/B07B4G5RBN.glb' --output data/abo500_single/B07B4G5RBN_ATVPDKIKX0DER/model.glb
python -m tactile3d.experiments.single_object \
  --root "$ABO_ROOT" \
  --mesh data/abo500_single/B07B4G5RBN_ATVPDKIKX0DER/model.glb \
  --config configs/single_object_chaise.yaml \
  --output outputs/abo500_single_reproduced --stage all
```

可将`--stage all`分为`predict`、`evaluate`、`plot`依次执行。evaluate会验证保存的预测与字段校验和，发生修改时拒绝评分。

本次相关自动化检查：12项通过，覆盖有限体积质量/惯量解析解、混合界、重叠唯一归属、先验积分一致、标签与预测隔离、产物修改检测，以及现有ABO读取/体积/统一状态功能。代码静态检查通过；四张图已逐张检查标注、图例、色标和裁切。

## 8. 到论文主实验还缺什么

1. 将对象特定的几何规则替换为可泛化的精细部件模型，使用train/val开发，保留独立测试对象。
2. 用真实Qwen结构化推断及训练/验证监督替换人工内部假设，检验密度—占用不可辨识与校准。
3. 接入InstantMesh，单列重建几何轨道；将原始Mesh、数据集alpha等额外信息预算列清楚。
4. 在同对象、同输入/训练预算下实际运行NeRF2Physics、PUGS、PhysVGGT。当前三者均未执行，本次图表没有冒用其论文平均数作为单例成绩。
5. 增加独立局部属性/内部构造参考，随后进行编辑配对与机械臂实验。

本次结果支持继续实施和扩大实验，尚不能支持达到T1水平或超越已有论文的结论。
