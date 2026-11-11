# 最新实验图件入口

这些文件对应同一轮躺椅开发实验。直接查看下表中的主要图即可，无须逐个打开文件。

| 内容 | 预览 | 可编辑版本 |
| --- | --- | --- |
| 接触点与支撑多边形 | [PNG](contact_geometry.png) | [SVG](contact_geometry.svg) |
| 材质和物性先验 | [PNG](dense_property_maps.png) | [SVG](dense_property_maps.svg) |
| QEM 减面对比 | [PNG](qem_mesh_comparison.png) | [SVG](qem_mesh_comparison.svg) |
| 匹配面数的精度与耗时 | [PNG](matched_accuracy_cost.png) | [SVG](matched_accuracy_cost.svg) |
| 局部换材及质量矩更新 | [PNG](local_material_edit.png) | [SVG](local_material_edit.svg) |
| Mesh、VLM 与融合消融 | [PNG](grounding_ablation.png) | [SVG](grounding_ablation.svg) |
| 方法流程 | [PNG](method_overview.png) | [SVG](method_overview.svg) |

`physical_parameter_sensitivity` 是补充敏感度图。
正式比较使用 `matched_accuracy_cost` 和匹配面数表。

文件分工如下。

- PNG 用于查看；SVG 用于排版。同名文件是同一张图。
- `source_data.json` 和 `matched_comparison.csv` 保存数值及统计口径。
- `part_state_geometry_constrained.json` 保存统一部件状态，`contact_geometry.json` 保存接触几何。
- `*_raw*.json` 和 `grounding_visual_evidence_answers.json` 保存推理记录，通常只在复现时查看。
- `CAPTIONS.md` 解释图例、单位、来源和评价范围；`artifact_validation.json` 保存文件校验信息。
- PDF、TIFF 是本地投稿导出版本，由相同脚本生成；GitHub 主要保留预览、SVG 和源数据。

本轮使用参考几何；物性字段是参数先验，质量矩计算限定于四条闭合木腿。
完整结果见[实验说明](../../docs/results/chaise_reference_20261111_zh.md)，
复现命令见[运行方法](../../docs/REFERENCE_MESH_STUDY_zh.md)。
