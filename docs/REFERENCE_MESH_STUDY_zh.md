# 参考几何上的分轮推断与任务约束减面

本实验固定使用 ABO 躺椅 B07B4G5RBN_ATVPDKIKX0DER。原始 RGB 和艺术家参考
模型是推理输入，用于隔离重建误差，检查部件构造推断、减面及材料编辑。
这不是图片重建基准。该对象虽源于 ABO 测试划分，但此前已用于开发，不能再作为盲测。

## 方法与实验协议

1. 焊接重合顶点，避免 UV 接缝被误认为部件断裂。保留纹理采样与原面顺序，
   按连通性产生候选区域。候选区域不直接等于语义部件。
2. 输入三张原始视图和编号候选区域图，用同一 Qwen 模型进行两组实验。
   同题重复组每次独立询问，分轮组依次进行整体判断、几何检查和冲突修订。
   三个随机试验，每组每试验三次调用。分轮问题选择依赖上一轮回答，不读取质量标签。
   记录全部提示、回答、token、随机种子、耗时、峰值 GPU 分配与格式失败。
   材料与构造分别聚合，构造票数并列时保留未知。可见材料以单人复核的粗粒度
   开发标注评价，不将其当作内部构造或物性真值。
3. 固定参考面数预算 2400、1200、600，比较全局 QEM、按连通区域保护的 QEM、
   薄结构与物理敏感度引导的区域预算分配。QEM 核心使用同一实现。
   敏感度是排序代理，尚无物理误差上界，也未实现逐边的接口保护。
4. 区域预算按容量限制分配整数面数，区域内执行 QEM，直接保留区域身份。
   全局 QEM 使用原三角面中心最近邻进行近似身份映射，映射耗时计入成本。
   报告实际面数，不能假定 QEM 严格达到请求面数。
5. 在 VLM 判为木材且参考网格闭合的区域上，以固定密度先验独立积分质量、
   质心和完整惯量。开放区域不输出物理误差，不用带符号体积冒充材料体积。
   比较一次局部铝材编辑的增量结果与全量重算残差。

## 图件约定

结构回答封存后，另做三次共享材料参数查询，明确木材模量方向、织物面内模量、
干燥不锈钢滑块接触条件与 Shore D 硬度量表。该阶段与两组结构对照分开计费。
保留 VLM 缺失输出，密度与模量可绑定现有材料库先验，逐字段记录来源；
共享参数查询的完整结果另外保存，不改写原始结构回答。

使用 Python 和 Matplotlib，导出 PNG、可编辑文字 SVG、PDF、600 dpi TIFF 与数值源数据。
图包括原始观测、参考渲染、部件材质、物性先验、减面可视化、误差与成本、局部编辑。
渲染使用统一相机、逐像素深度测试与插值法向。未知属性保留灰色。
密度和模量图表达材料先验，不能代表已恢复软包内部。没有相容硬度量表或指定摩擦接触对时，
相应参数不参与汇总。木架物理量与整物体物理量分别命名。

## 运行入口

```bash
python scripts/run_reference_mesh_study.py prepare --output outputs/chaise_reference_20261111
python scripts/run_reference_mesh_study.py vlm --output outputs/chaise_reference_20261111
python scripts/run_reference_mesh_study.py finalize --output outputs/chaise_reference_20261111
python scripts/evaluate_reference_vlm.py --source outputs/chaise_reference_20261111
python scripts/query_reference_material_priors.py --source outputs/chaise_reference_20261111
python scripts/run_reference_mesh_study.py benchmark --output outputs/chaise_reference_20261111
python scripts/run_reference_mesh_study.py figures --output outputs/chaise_reference_20261111
```

代码模块分别为 `vision/progressive.py`、`mesh/task_lod.py`、`figures/mesh_fields.py`。
新输出目录禁止覆盖，VLM 原始结果先封存，再进行几何和编辑评价。
本轮原型检验的是参考几何条件下的机制，不声称减少首次重建成本，也不声称物性实测准确性。
