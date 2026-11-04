# Part Token 最小验证

真实输入：官方 ABO 沙发 B075X4QMX3 的商品图和 GLB，与上一次 pilot 使用相同资产。尚未确认该物品属于 ABO-500 子集。

```bash
pip install -e '.[figures]'
# 如果本地尚无资产，先运行 tactile3d abo-pilot 下载官方资产。
tactile3d token-mapping-pilot --asset-dir data/pilot/B075X4QMX3 --output outputs/part_token_pilot --pitch 0.04 --samples 4000
```

输出三组 PNG / SVG / PDF：

1. `01_part_token_fields`：商品图、部件、密度、E、nu、占用率、部件质量、密度不确定性。
2. `02_mass_comparison`：全布实心、部件材料混合实心、分层占用先验三种假设的质量比较，以及区间和部件质量。
3. `03_mapping_consistency`：几何种子、几何图优化和低面数映射；质量、COM、完整惯量张量的跨LOD差异。

数据包含 `part_tokens.json`、`canonical_state.npz`、`part_parameters.csv`、`mass_metrics.csv`、`lod_metrics.csv` 与 `report.json`。

## 如何解释

- 该版本实现显式、稳定的部件状态，尚未训练视觉/几何/材料投影层，也未运行 SAM、VLM 或 InstantMesh。
- 分割来自沙发几何位置种子、法向加权图扩散和连通片投票，没有人工部件真值。改动面积比例不能视为精度提升。
- 原始 GLB 非水密，采用固定物体坐标系中的体素化与孔洞填充构建包络代理；这不是内部材料的真实测量。占用率只表达材料填充假设。
- 内部材料比例及属性沿用旧沙发 pilot 的人工先验，没有利用当前物品质量调整先验。区间未经独立验证集校准。
- E、nu 为示意性混合描述量，并非通过力学均匀化或实测得到的等效弹性参数。图中的质量颜色是部件总质量，不是面密度。
- 不同LOD沿用同一密度向量。对照为各LOD独立重体素化后积分；统一路径保留原始体积单元，因此零漂移是设计保证，仅验证数据流一致性，不证明更准确的物理推断或通用守恒重映射算法。
- COM和惯量只报告相对参考体素状态的漂移，没有真实COM或惯量标注。惯量使用完整张量差的Frobenius范数，积分考虑体素自身惯量。
- 简单质量对照不是 PUGS、VoMP 的复现结果；单物体不能代替独立测试集。
