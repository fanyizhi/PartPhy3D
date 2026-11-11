# v0.6.4 实际实验记录

2026-11-07，Linux/Python3.10、RTX4090 24GB，使用已有固定五物体以及官方训练划分。
“实际执行”指程序确实运行，不表示做过实物称重、材料测量或机器人实验。
[方法、公式和完整命令](../../METHODS_V064_zh.md)。

## 数据证据

本地 `D:/06-数据集/abo_500` 的500对象、15000图、300/100/100划分和质量标签均已
扫描核对；目录内没有明确的商品尺寸字段，材料与厚度字段不作为内部真值。
[全目录审计](full_abo500_evidence_audit.json)。

另从 [ABO官方商品元数据](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
匹配全部500对象的尺寸，使用 `item_dimensions` 的已知单位转换，排除重量和包装尺寸。
归档校验值、逐条记录哈希和市场匹配保留在
[500对象尺寸输入](../../../configs/evidence/abo500_catalog_dimensions.json)。
商品字段可能有轴名、尺寸或标注误差；仅使用最大边长的辅助归一化，不声称独立米制标定。

## 生成阶段稀疏解码

[固定生成前端的稀疏计算对照 (historical)](https://github.com/fanyizhi/PartPhy3D/blob/a6d87d5399dff9979565d084b6001af8bb796e3a/docs/results/geometry_physics_20261107/sparse_decoder_benchmark.png)

| 对象 | 实际运行的单元权重比例 | 解码/提取阶段中位提速 |
| --- | ---: | ---: |
| 躺椅 | 1.22% | 4.59× |
| USB集线器 | 0.93% | 4.66× |
| 梯子 | 0.43% | 4.79× |
| 镜子 | 0.47% | 4.73× |
| 保护套 | 1.19% | 4.62× |

相同三平面、相同分辨率，暖机一次后交替重复3次。五个物体均为同一三角面拓扑，
最大顶点差不超过2.98×10⁻⁷个模型单位。单元比例针对权重 MLP，不是全部 SDF 查询。
保留2146689个SDF顶点查询，尚未改变三角面数量或实现接口/物理感知的构建预算。
这也不是整个系统4.6倍的端到端加速。
[原始重复计时和计数](sparse_decoder.json)、
[体积、COM和单位密度惯量差异](sparse_geometry_moments.json)。

稀疏选项已接入实际前端，并对保护套重新生成带顶点色的50148面水密 Mesh；
24个未输入视角轮廓 IoU 为0.7251226，仍有原始几何不足，没有用提速掩盖误差。
[真实前端输出](sparse_frontend_gate.json)、[留出评价](sparse_frontend_evaluation.json)。
全量生成结果是数值参照，不是几何真值。

## 从图片学习尺度

[尺度学习、误差和独立校准区间 (historical)](https://github.com/fanyizhi/PartPhy3D/blob/a6d87d5399dff9979565d084b6001af8bb796e3a/docs/results/geometry_physics_20261107/image_scale.png)

冻结CLIP图像特征、对数长度岭回归；300训练对象，50验证对象选正则参数，另50对象
独立校准，100测试对象。最终alpha=1.0。测试质量没有输入训练，测试尺寸不参与选参。
重复执行得到逐字节一致的测试预测。

| 方法 | 最大边长 MAE / m ↓ | APE / % ↓ | 对数绝对误差 ↓ | 1.25倍范围内比例 ↑ |
| --- | ---: | ---: | ---: | ---: |
| 训练集几何均值 | 0.5940 | 84.17 | 0.6689 | 28% |
| CLIP＋岭回归 | 0.3115 | 36.69 | 0.3649 | 43% |

排除历史开发的10个对象后，余90个的学习尺度 APE 为36.98%。这并非新选的独立最终
方法测试集；保留完整100和补充90的结果。[报告](image_scale_report.json)、
[参数选择](image_scale_validation.json)、[四类对象ID](image_scale_split_roles.json)、
[封存预测](image_scale_test_predictions.json)、[协议](image_scale_protocol.json)。

名义90%的尺度区间在100测试对象中覆盖90个，平均宽度1.3203m；区间较宽，
不是高精度尺寸标定，也不是全物理参数的90%区间。只观测到一次数据划分，
不能凭90/100的覆盖数保证新分布同样覆盖。
使用标准 [split-conformal方法](https://arxiv.org/abs/2107.07511)，不将此技术本身列为原创。

## 部件闭合域、材料编辑及条件质量

[商品尺度辅助下的局部薄层质量 (historical)](https://github.com/fanyizhi/PartPhy3D/blob/a6d87d5399dff9979565d084b6001af8bb796e3a/docs/results/geometry_physics_20261107/catalog_layer_masses.png)

固定部件：躺椅坐垫表皮、集线器顶部、梯子顶部平台、镜面、保护套皮革片。
按公开的厚度/最大边长比0.001、0.002、0.005构造，不按质量挑厚度。
采用方向分片、必要时逐三角柱并集与外包域交集；保持孔洞，避免重叠材料重复计质。

15个设置中13个通过，躺椅两个设置因边界积分与Manifold体积核对不一致被拒绝，
没有放宽检查强行通过。[完整接受/失败记录](catalog_patch_report.json)、
[局部数值CSV](catalog_layer_masses.csv)。中间厚度0.002下五物体均有有效局部域。
换材后复用几何矩，核对质量、质心和完整惯量的全量重新积分。

图片学习尺度也已接入上述路径：
[使用图片尺度的局部薄层质量 (historical)](https://github.com/fanyizhi/PartPhy3D/blob/a6d87d5399dff9979565d084b6001af8bb796e3a/docs/results/geometry_physics_20261107/learned_scale_layer_masses.png)、
[记录及失败](learned_scale_patch_report.json)、[CSV与尺度传播范围](learned_scale_layer_masses.csv)。
此时测试物体尺寸只用于尺度模块的独立评价，不输入局部域的构造。

**以上均是所选局部薄层的条件质量，不能与商品整物体重量直接计算APE。**
其它部件、框架、填充、空腔和未观察区域仍未知。薄层厚度及材料值也不是测量。
任何由尺度区间立方传播的质量范围，都只反映固定构造下的尺度不确定性，
不是经过真实质量校准的完整物理区间。

## 刚体链路与综合入口

[相同冲量下局部条件材料域的响应 (historical)](https://github.com/fanyizhi/PartPhy3D/blob/a6d87d5399dff9979565d084b6001af8bb796e3a/docs/results/geometry_physics_20261107/conditional_rigid_response.png)

上述图为商品尺寸辅助的条件片；[图片尺度下的响应 (historical)](https://github.com/fanyizhi/PartPhy3D/blob/a6d87d5399dff9979565d084b6001af8bb796e3a/docs/results/geometry_physics_20261107/learned_scale_rigid_response.png)
采用同一冲量设置。程序检查能量、世界角动量及四元数范数漂移。
这是零重力、无接触的局部刚体数值诊断，不验证 E、摩擦、硬度、变形或实物交互精度。
[数值记录](rigid_response.json)、[图片尺度记录](learned_scale_rigid_response.json)。

七阶段串行入口已实际执行单个对象：定位、SAM2、点位材料、冲突拒绝、表面编辑图、
局部闭合域、刚体响应，逐阶段保留日志；[综合验收记录](integrated_gate.json)。
五物体各模块均已单独执行，但不把单物体综合耗时直接外推为全数据集耗时。

最新入口另用图片预测尺度重新执行躺椅，7阶段均正常结束，阶段耗时合计211.64秒。
局部构造的3个厚度设置仅1个通过，另外2个仍因几何核对失败而保留拒绝记录；
流程正常结束不代表全部几何候选均成功。
[图片尺度综合记录](learned_scale_integrated_gate.json)。
[Windows与Linux各205项测试、依赖及数据检查](verification.json)只证明软件验收，
不代替物理精度与论文贡献的实验验证。

## 独立分割评价准备

[人工标注页面核验截图 (historical)](https://github.com/fanyizhi/PartPhy3D/blob/a6d87d5399dff9979565d084b6001af8bb796e3a/docs/results/geometry_physics_20261107/annotation_ui_qa.png)。已实际打开页面验证画笔、撤销、
切换案例和PNG导出；导出测试保持0/255，未创建人工语义真值，也未勾选人工核验。
[功能核验记录](annotation_ui_qa.json)。

15张独立人工部件标签尚缺，分割指标保持空：
[待标注报告](part_annotation_pending.json)。默认评价 Mesh 标签的可见投影，
二维提议可单独评价；不让模型给自己的掩码当真值。
官方部件挑战只覆盖chair/table/cabinet/lamp/bed五类，不能假定它提供当前五个产品
的全部细粒度标注，见[官方挑战说明](https://3dv-in-ecommerce.github.io/iccv2023/)。

## 完整方法目前不能报告什么

完整 PartPhy3D 质量、局部物性准确率和真实编辑后质量仍无有效分数；
PUGS、PhysVGGT、Gen2Physics的复现没有在本轮完成。先前已运行的
[NeRF2Physics缓存质量对照](../surface_physics_20261107/README_zh.md)保持原始结果，
尺度误差不能代替质量误差，局部片质量不能代替整物体质量。
共享潜空间训练、完整内外结构推断、接口/物理感知的生成预算以及接触/机器人验证
仍需继续研究。软件架构和数值检查通过不等于这些论文结论已经成立。
