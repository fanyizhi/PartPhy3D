# 对比实验：已实现入口、当前结果与待完成项目

日期：2026-11-03。入口为`python scripts/compare_experiments.py`。本次仅本机CPU评价和绘图，未启动训练或操作服务器队列。

## 1. 已实现的固定清单评价

- 新入口读取注册表中的固定对象清单，不按各方法成功对象的交集自动改变主表分母。
- `executed`且完整覆盖清单的方法计算指标；缺失为`incomplete`，错误为`invalid`，未运行为`pending`。后三种主表指标为空，不填0，不填论文平均分。
- 记录输入条件、来源文件与实际运行证据文件的SHA256。它们支持追溯，不自动证明外部文件真实、公平或无泄漏；复现时仍需保存命令、模型/代码版本、退出状态和输入预算。
- 统一逐物体ADE、ALDE、APE、MnRE、RMSE；APE在JSON原始误差中是比例，在主CSV表`ape_percent`和图中是百分数。
- 支持同对象配对bootstrap区间与方法差异。固定清单为当前10例开发集，不据此宣称最终未见测试优势。
- 同时导出表格、逐物体源数据、指标图、预测—参考散点图、误差分布与已报告分位区间诊断图，PNG/SVG/PDF。

2026-11-13更新：`paper-figures`命令必须提供`--cohort`，不能自动取交集。Python接口的交集模式仅保留给探索分析；正式对照传`cohort_ids`或使用注册表入口，保持固定清单。

## 2. 修正范围与分位数的语义

原生`[low, high]`只证明给出了范围。新导入器取中点作为该协议下点预测，将端点写为`range_low/range_high`，不再伪装成`q05/q95`或中位数。显式q05/q95另行保留；分位数端点本身不能定义均值，必须提供独立点预测。

所有质量必须有限且为正，不能用epsilon把负数、零或NaN伪装为有效预测。同一物体多个视角的结果不自动平均成一个物体预测；需要明确聚合协议，否则拒绝重复映射。

报告的q05/q95可以来自未校准先验；命中比例不自动是后验校准证据。原生范围不进入分位数覆盖评价。

## 3. 本次实际结果

来源为`outputs/abo500_ten_v2`封存预测。本次重新统一评价，没有重新生成Mesh、调用Qwen或学习部件模型，没有改变参数或预测。当前新结果目录为`outputs/comparison_ten_development_v3`。

| 方法 | 物体数 | ADE kg ↓ | ALDE ↓ | APE % ↓ | MnRE ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Surface solid | 10 | 22.5201 | 1.3292 | 283.6399 | 0.3351 |
| Shell only | 10 | 7.6100 | 0.8687 | 67.8053 | 0.5072 |
| Construction prior | 10 | 11.4145 | 0.7577 | 88.8394 | 0.4900 |

以上为原始参考Mesh、缓存语言建议与规则构造的对照，不是学习式完整PartPhy3D，也不是以下论文的复现。构造先验的ALDE较低，但ADE、APE与MnRE没有全部优于仅壳层，对失败与重物体误差必须保留。

注册表中的NeRF2Physics、PUGS、PhysVGGT、Gen2Physics、VoMP及学习式完整PartPhy3D共6行保持`pending`，没有成绩。前3个是既定主要质量对照，后2个按可复现性、输入及属性协议决定主表或补充表。

输出：

- `registry.json`：同一物体清单、目标来源和方法状态，可接入真实复现结果。
- `mass_comparison.csv`、`mass_comparison.md`：完整主表，包括未运行/失败状态。
- `benchmark_report.json`：来源哈希、缺失情况、数值与图表清单。
- `figures/comparison_source_data.csv`：逐方法逐物体的源数据。
- `figures/mass_metrics_comparison`、`predicted_vs_ground_truth`、`per_object_ape`和`uncertainty_calibration`：PNG/SVG/PDF。最后一个保留历史文件名，内容标明只是报告的分位数诊断，不宣称校准成功。

本轮24项相关测试通过，覆盖固定队列、缺失和标签冲突、原生范围/分位数区分、NaN/零/负质量拒绝、重复视角映射、封存路径、绘图、既有ABO/pipeline及局部属性评价。不完整或无效方法会使命令行返回非零状态，待运行方法不当作执行失败。

## 4. 如何复现及接入真实论文结果

在项目根目录使用现有依赖环境：

```bash
python scripts/compare_experiments.py
```

默认输出到`outputs/comparisons/run_<编号>`，重复运行不覆盖旧结果。

显式目录：

```bash
python scripts/compare_experiments.py \
  --pilot outputs/abo500_ten_v2 \
  --output outputs/my_comparison
```

论文代码实际跑完后，先规范化其逐物体结果，例如原生标量/范围映射：

```bash
python -m tactile3d.cli import-mass-baseline \
  --predictions results/pugs_native.json \
  --targets outputs/my_comparison/targets.json \
  --method PUGS \
  --output results/pugs_normalized.json
```

以上`results`文件仅是调用格式示意，当前并不存在真实PUGS预测。不能将论文中汇总数字写进这些文件。

在生成的注册表中，将对应方法改为`executed`，提供相对于注册表或绝对的`predictions_path`及`evidence_path`，声明真实输入条件，然后运行：

```bash
python scripts/compare_experiments.py \
  --registry outputs/my_comparison/registry.json \
  --output outputs/my_comparison_with_baselines
```

新输出目录中会冻结本次规范化结果并记录原文件哈希，注册表不被评价程序改写。提供完整结果文件才能进入同队列主表。

## 5. 还需要的对比，不因入口先做质量而遗失

| 实验 | 对照/指标 | 当前证据 |
| --- | --- | --- |
| 原物体质量主表 | NeRF2Physics、PUGS、PhysVGGT等；ADE/ALDE/APE/MnRE/RMSE | 统一入口已实现，论文复现待运行 |
| 局部物理属性 | 对应实测协议下密度、摩擦、硬度、模量等误差 | 局部指标接口已有，当前10例实测标签缺失，不填预测当真值 |
| 编辑效果 | 仅材质、仅参数、构造同步、接口约束；独立编辑后质量/COM/惯量 | 解析状态编辑已验证，真实效果对比待做 |
| 稀疏化效率 | 原网格、普通QEM、任务约束；面数、总耗时、内存和物理/接口误差 | 普通QEM诊断已有，任务约束方法及真实加速待验证 |
| 机制消融 | 去内部构造、去共享状态耦合、去物理/接口稀疏约束 | 待核心模型实现后同预算运行，不预填提升数值 |
| 部件/边界 | 部件/材料IoU、Boundary F-score、连接局部检查 | 需独立参考分割，当前几何聚类不是真值 |
| 交互 | 相同控制/外力与固定参考环境下的响应、滑移等 | 仿真/实物后续阶段，未运行 |

官方Mesh诊断与InstantMesh生成Mesh的主流程分轨；同数据集不等于同输入条件。公开论文若只有不同对象集合/输入预算的汇总分，只作单独文献参考，不能混入本程序的同对象主表。

当前10物体已参与开发；扩展全数据集时仍保留训练/验证/独立测试边界，不以开发集成绩替代泛化结论。
