# 真实生成 Mesh 上的部件换材重放

这是已执行推理结果的 CPU 重放，不会重新调用 Qwen、SAM2 或 InstantMesh。
包含 ABO 物品 B013TGFGWS_A1F83G8C2ARO7P 的生成 Mesh、三视角融合标签和
第 00 视角掩码。使用六张真实观测图生成 Mesh，使用 00/10/20 三视角做部件分析；
没有使用参考 GLB 作为生成结果。源数据遵循仓库 `data/abo500_five/ATTRIBUTION.md`。

在项目根目录执行：

```bash
python -m pip install -r requirements.txt
python scripts/render_surface_maps.py \
  --fields examples/observed_surface_demo \
  --edit-spec examples/observed_surface_demo/edit_spec.json \
  --output outputs/observed_surface_replay
```

默认把包上的 `leather_patch` 改成 `steel`，同时更新颜色、密度和杨氏模量先验。
输出包含 PNG/SVG、前后 PLY、前后 Part Token、逐面 NPZ、参数 CSV 和编辑记录。
脚本会检查非目标面的颜色、密度、杨氏模量及非目标像素没有改变。
输出目录必须不存在，避免覆盖实验。

这不是钢材真实外观生成，也没有建立该小部件的封闭材料域与米制尺度，因此
**不会报告该真实物品编辑后的质量、质心或惯量**。灰色区域保持未知。
三视角投票一致性不是校准概率；材料及物性是条件假设，不是实测真值。
选这个例子是为了检查小区域的局部编辑，全部五个物体的好坏结果均见
`docs/results/surface_physics_20261107/`，不能只用这个例子报告精度。
