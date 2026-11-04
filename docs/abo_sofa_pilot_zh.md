# ABO 单物体论文图复现

本 pilot 使用 ABO 官方物品 `B075X4QMX3`。程序会下载商品图和官方 GLB，生成物理约束减面、分层内部构造后验、部件物理场及质量不确定性图。

```bash
conda activate partphy3d
pip install -e .
tactile3d abo-pilot \
  --asset-dir data/pilot/B075X4QMX3 \
  --output outputs/abo_sofa_pilot \
  --samples 8000
```

主要输出：

- `partphy3d_pipeline.png`：完整方法 pipeline。
- `abo_sofa_physical_fields.png`：部件、密度、杨氏模量、泊松比、实心率图。
- `abo_sofa_mass_and_lod.png`：总质量后验、部件质量、材料质量和减面误差图。
- `abo_sofa_mesh_lod.png`：原始 Mesh 与物理约束 LOD 对比。
- `pilot_report.json`、`pilot_source_data.csv`：全部图的可审计源数据。

注意：这是一个真实 ABO 资产上的 construction-prior pilot。商品真实质量只用于画图评价，没有用于该物品的后验推断；VLM 微调结果必须在 ABO-500 的独立训练/验证/测试划分上报告，不能把该 pilot 当成完整 benchmark。
