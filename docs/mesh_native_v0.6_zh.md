# Mesh原生主线：实现、复现与验收

2026-11-05，v0.6.0。本版把已知构造的物理积分、材料编辑及受控响应学习接入
**封闭材料区域Mesh**，不调用体素化。它完成一个必要的数学与软件环节，尚不是
真实图像到精细部件、隐藏构造推断的完整模型。旧体素原型作为对照保留。

## 1. 安装和第一条复现命令

Linux终端，项目根目录，建议Python 3.10：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,figures]"
python -m pytest -q
python scripts/reproduce_mesh_native.py --output outputs/mesh_first
```

不需要torch、数据下载、4090或SSH。默认384个合成诊断对象、350轮、3个
初始化；真实开发队列依然是5个ABO物体，二者不是同一个实验。
不要向共享的训练环境直接执行依赖升级。所有结果必须使用新目录。

```text
outputs/mesh_first/
  suite.json                          总状态、范围、耗时、未完成项
  ARTIFACTS_SHA256.json                结果文件哈希
  editing/
    original/asset.json                原始区域、材料、单位、版本
    original/region_000.npz            三角面与米制顶点，不是体素
    original/part_tokens.json          部件聚合数据；不是已训练潜变量
    original/display.ply               示意颜色Mesh
    edited/                           编辑后的同格式资产
    edit_spec.json                    可重放的材料编辑指令
    report.json                       解析核对、重算核对、作用范围
    physics.csv                       质量、质心、完整惯量张量
    regions.csv                       前后各区域的密度、填充率、E、ν
    mesh_edit_views.png                三个视角换材前后对照，另有SVG
    mesh_property_sections.png         外壳/填充的ρ、填充率、E截面
    mesh_physics.png                   质量、质心、对角惯量，另有SVG
  learning/
    dataset/                          输入、独立解析标签分文件
    mesh_geometry_basis.npz            从Mesh积分得到的候选体积
    seed_17/                          检查点、预测封存、MAP资产与编辑
    metrics.csv                       ADE、ALDE、APE、MnRE
    test_predictions.csv              测试预测和校准区间
    training.csv                      实际训练日志
    learning_results.png              学习和误差图，另有SVG
```

单独运行积分/编辑或学习：

```bash
python scripts/run_mesh_native.py --output outputs/mesh_edit
python scripts/train_mesh_state.py --output outputs/mesh_learn
```

## 2. 现在真正执行的计算

一个部件含多个**互不重叠的正体积材料域**。外皮与内部框架、泡沫分别建域，
每域记录材料、来源、宏观填充比例、几何、版本。表面分割得到的开放面片
不能直接当成闭合材料域；外表贴上“皮革”也不能把整个内部体积赋成皮革。

代码：`physics/mesh_native.py`、`mesh_asset_io.py`。

1. 检查有限顶点、合法面、闭合性、定向一致性及正有向体积。
2. 显式外壳通过双精度Manifold Mesh差集建立，空腔边界保持反向定向。
3. 每对区域用包围盒预筛，再以Mesh交集检查正体积重叠；接触界面可以共享。
4. 对有向三角形构成的四面体直接积分，得到区域体积、体积质心与中心二阶矩。
5. 用区域有效密度和移轴定理合并质量、世界质心和完整惯量张量。

记区域为Ωᵣ，材料密度ρᵣ，宏观填充比例fᵣ：

\[
m_r=\rho_r f_r V_r,\quad
c=\frac{\sum_r m_r c_r}{\sum_r m_r}.
\]

区域中心二阶矩为 \(Q_r=\int_{\Omega_r}(x-c_r)(x-c_r)^T dx\)。令
\(d_r=c_r-c\)，则

\[
S=\sum_r\left[\rho_r f_r Q_r+m_r d_rd_r^T\right],\qquad
I=\operatorname{tr}(S)I_3-S.
\]

积分在物体附近的参考点进行，降低大世界坐标下的数值抵消。缓存几何矩，
换材不重做三角面积分；区域数量上的合并运算仍然存在，不声称总流程O(1)。

此处f是域内**均匀有效占用**，不能恢复孔洞位置，也不是材料概率。
泡沫的实测体积密度通常已包含微观孔隙，不能再凭想象重复乘一个孔隙折扣。
逐区域常数E、ν是材料参数记录，不等于已验证的装配刚度、抓取摩擦或关节强度。

输入需要几何有效。拓扑和布尔库检查不构成对任意自相交输入的独立完备证明。
不自动封洞、取凸包或把负体积取绝对值；极小体积阈值见文件头。

## 3. 默认参考及编辑

默认外盒0.20×0.30×0.40m，内部0.16×0.26×0.36m，平移到
(0.10, −0.04, 0.20)m。外壳ρ=1000kg/m³，内部泡沫ρ=50kg/m³。
外壳替换为ρ=2700kg/m³、E=69GPa的指定材料，内部和几何保持不变。

解析参考：外壳0.009024m³，内部0.014976m³；总质量
**9.7728→25.1136kg**。参考由盒体积与盒惯量公式独立计算，不再调用积分器。
这是指定构造核对，不是从图片成功预测真实物体质量。

颜色、密度和弹性属性来自同一个Material对象。渲染是固定光照的不透明示意图，
不是PBR真实材质。零非目标像素变化仅适用于该渲染器，不能推广到反射、阴影场景。
截面由Mesh平面切片形成多边形，不建立三维体素网格。斜线灰色表示E未知。

重放一次编辑：

```bash
python scripts/run_mesh_native.py \
  --asset outputs/mesh_edit/original \
  --plan outputs/mesh_edit/edit_spec.json \
  --output outputs/mesh_replay
```

材料编辑计划的完整例子见`examples/mesh_material_edit.json`。编辑只改指定区域，
要求版本匹配；不会偷偷改变壁厚或填充比例来补偿质量。

## 4. 接入自己生成或标注的区域Mesh

先准备真实尺度、闭合、有依据的区域。以下只是格式，不提供凭空推断的内部真值：

```json
{
  "object_id": "my_object",
  "mesh_unit_to_meter": 0.001,
  "regions": [
    {
      "part_id": "body",
      "region_id": "shell",
      "outer_mesh": "outer_mm.ply",
      "cavity_mesh": "cavity_mm.ply",
      "provenance": "measured_CAD_or_explicit_hypothesis",
      "fill_fraction": 1.0,
      "material": {
        "name": "specified_polymer",
        "density_kg_m3": 1000,
        "source": "specified_reference",
        "display_rgba": [175, 125, 75, 255]
      }
    }
  ]
}
```

路径相对该JSON；若有填充，再增加一个区域，其outer_mesh使用内部闭合Mesh。
`mesh_unit_to_meter=0.001`仅适用于毫米输入。未知尺度不能靠猜测设成1。

```bash
python scripts/run_mesh_native.py --input /data/my_regions.json \
  --plan examples/mesh_material_edit.json --output outputs/my_mesh_edit
```

只有表面标签时程序会拒绝，必须先建立材料体域。VLM候选需要保持“假设”来源，
不得写成测量。当前不自动生成皮革/泡沫/木架的真实内部几何。

## 5. 可微物理学习及其证据边界

`research/mesh_templates.py`用Mesh差集生成实心、闭壳、壳＋填充、框架＋填充。
`native_learning.py`把积分体积作为可微模型中的几何基底。几何布尔运算本身
不参与反向传播；本版学习的是候选概率、密度修正和宏观填充比例。

\[
\widehat m^{(e)}=\sum_k p_k\sum_r V_{kr}f_r\rho_r^{(e)}.
\]

沿用已实现的Huber对数质量损失、带符号的归一化编辑质量变化损失和0.15倍
构造分类交叉熵。编辑参考来自独立解析设计，不能用模型自己更新出来的结果
监督自己。对照为质量监督模型、增加编辑响应监督模型及直接响应回归模型；
三者使用同一族划分和配对训练标签。编辑响应监督不是唯一可辨识隐藏构造的证明。

预测在评测前写入NPZ并记录SHA-256。训练246、校准66、测试72，按物体族
隔离。同一划分三种初始化是优化波动，不是三个独立数据集。报告原物体、已见
编辑和未见钢材编辑的误差；混合期望与MAP可导出资产分别评测，不能混用分数。
校准的是质量区间，不是内部构造后验已经校准。

输入仍是11维合成观测代理，不是真实RGB/Mesh编码器，也不是千问微调。
小MLP用于先检查机制能否训练；不能把本实验称为ABO领先结果或PUGS复现。

## 6. 与完整研究方法的衔接

| 环节 | 本版完成 | 下一验收条件 |
| --- | --- | --- |
| 图像→外表Mesh | 保留InstantMesh适配器 | 在真实图像运行权重，核验尺度和相机；本版未执行 |
| 精细部件与连接 | 输入接受明确部件/区域 | 多视角语义＋几何对应，独立部件标签和接口误差 |
| 内部构造推断 | 显式候选可执行 | VLM提出候选；测量/拆解/CAD约束，未知不编造 |
| 数据统一 | 部件→区域→几何/材料/物理量的可重放schema | RGB/Mesh/材料投影与融合编码器另行训练、消融 |
| 物理准确性 | 独立解析核对和受控学习 | 封存5物体预测，再读独立称重；E、μ等各自需要真值 |
| 材料编辑 | 颜色、ρ、E、ν同步；质量/COM/I重算 | 真实分割区域绑定、编辑前后独立参考 |
| 稀疏几何构造 | 旧解析场细分诊断保留 | 真实解码场内分配预算＋误差门槛，尚未接入 |

稀疏构造的目标是**生成/最终曲面构造阶段分配计算**：部件边界、薄壁、连接和
物理敏感区域保留采样，平坦低敏感域减少采样。不能只生成密Mesh后QEM减面就
把它作为该贡献。QEM保留为对照。当前任务加权解析细分的全局体积误差较大，
不能作为已达标提速；后续同时检验几何、体积、质量/COM/I、接口和总资源成本。
体积采样场即便由上游解码器使用，也应与本物理层是否使用体素分开报告。

千问训练分两阶段：先以专业标注训练结构化材料/构造候选输出（LoRA/QLoRA），
再训练可微物理头/共享状态。通用SFT输出JSON的交叉熵不能自动获得质量积分梯度。
`training/dataset.py`现增加同物体显式train/test分割冲突拒绝；目标仍需人工审核，
VLM口头概率不当作实测置信度。5个开发物体不用于训练再宣称独立测试。

## 7. 依赖与验收

新增核心依赖`manifold3d>=3.5.0`；沿用numpy、trimesh、scipy，绘图为matplotlib。
全部依赖见根目录`requirements.txt`和`pyproject.toml`。Manifold为上游布尔运算库，
不是本项目原创算法。

测试覆盖独立盒参考、空腔减法、接触/重叠、非凸包含检查、任意姿态与大坐标、
完整非对角惯量、宏观填充不变、版本冲突、缓存不可变、文件篡改、米制转换、
四类Mesh基底、有限差分梯度、逐像素绑定和重放。
另将体素化、候选体素掩码和体素部件构造入口全部设为抛错，再运行新计算和
学习路径，以检查其没有依赖旧体素状态。新库依赖和Linux结果由对应提交CI核验。

本文档描述实现边界；不预先承诺优于论文、不保证期刊接收。
