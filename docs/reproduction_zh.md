# PartPhy3D 复现指南

本文给出从代码自检到 ABO-500 正式实验、VLM 推理、QLoRA 微调和论文出图的固定流程。

## 0. 当前可复现边界

当前仓库完整实现的是：

```text
metric textured mesh + multi-view images
  -> mesh 修复与部件分割
  -> 多视角材料证据与 VLM 内部结构候选
  -> 体积/壳体/填充质量积分
  -> 训练集质量校准
  -> ADE/ALDE/APE/MnRE 与不确定性评测
  -> 论文图和部件物理属性图
```

当前尚未集成“RGB 自动重建 mesh”的训练模型。正式实验有两条合法路线：

1. 使用与 ABO-500 场景配准且尺度已知的外部 mesh，复现当前 mesh-physics 主线。
2. 另接 image-to-mesh 模型，并单独报告其几何、尺度和物理误差，不能把外部真值 mesh 的
   结果写成从 RGB 端到端获得。

## 1. Linux 环境

推荐 Ubuntu 22.04、Python 3.10、NVIDIA 驱动、CUDA 12.1、两张 RTX 4090 24 GB。将项目
同步到服务器后，在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate partphy3d
pytest -q
```

通过标准：全部测试通过，`tactile3d --help` 能看到 `abo500-run`、`train-vlm`、
`paper-figures` 和 `object-figures`。

为了严格复现，每次实验保存软件和硬件环境：

```bash
mkdir -p outputs/reproducibility
python --version > outputs/reproducibility/environment.txt
nvidia-smi >> outputs/reproducibility/environment.txt
python -m pip freeze > outputs/reproducibility/pip-freeze.txt
git rev-parse HEAD > outputs/reproducibility/git-commit.txt || \
  echo UNCOMMITTED > outputs/reproducibility/git-commit.txt
```

当前仓库可能尚未建立首个 Git commit；正式服务器实验前应先提交固定版本，否则无法精确
追踪运行的代码。

## 2. CPU smoke test

这一步不下载大模型，用五个合成物体检查数据读取、mesh 修复、体积分、质量校准和评测：

```bash
bash scripts/run_abo500_smoke.sh
cat outputs/abo500_smoke/run/test/metrics.json
```

应生成：

```text
outputs/abo500_smoke/run/train/features.jsonl
outputs/abo500_smoke/run/mass_calibration.json
outputs/abo500_smoke/run/test/predictions.json
outputs/abo500_smoke/run/test/metrics.json
```

Smoke 数值是人工合成检查，不是论文结果。

## 3. 准备 ABO-500

下载公开发布数据：

```bash
bash scripts/download_abo500.sh data
```

期望目录：

```text
data/abo_500/
  filtered_product_weights.json
  splits.json
  scenes/<scene_id>/images/
  scenes/<scene_id>/transforms.json
```

你记得的 ABO mesh 确实存在：原始 ABO 发布包含 7,900 多个带纹理的 glTF/GLB 模型。当前
代码可以直接读取 `.glb`，无需先转换成 PLY。但是 NeRF2Physics 的 ABO-500 说明主要列出
图像、相机、质量、深度和点云，没有给出本项目可直接使用的一对一 GLB 目录。应通过原始
ABO 元数据中的 `3dmodel_id` 核验匹配后整理为：

```text
data/abo_500_meshes/<scene_id>.glb
```

也可以使用 `.ply`、`.obj`、`.stl` 或 `.off`。不能只截取 ABO-500 目录名中的 ASIN 自动
匹配，因为 NeRF2Physics 的公开 issue 已报告个别 ABO-500 场景标签与原 ABO 同 ASIN 商品
不一致。每个匹配都应至少渲染检查轮廓、朝向和纹理。

mesh 必须满足：

- 与 `transforms.json` 使用同一世界坐标系。
- 长度单位已知，例如一单位等于 1 mm 时 `MESH_UNIT_TO_METER=0.001`。
- 尽量闭合、法向一致、没有严重自相交。
- scene ID 与官方 split 完全一致。
- 原始 ABO `3dmodel_id`、ABO-500 `scene_id` 和本地 mesh 路径的映射保存成清单。

先只验证两个物体：

```bash
tactile3d abo500-validate \
  --root data/abo_500 \
  --mesh-root data/abo_500_meshes \
  --mesh-unit-to-meter 0.001 \
  --split test \
  --limit 2
```

mesh 或真实尺度不满足上述条件时，不应开始正式质量实验。

## 4. All-solid mesh baseline

`configs/abo500_mesh.yaml` 使用 OpenCLIP 表面材料证据和 all-solid 内部假设。先用两个样本做
GPU 集成测试：

```bash
tactile3d abo500-run \
  --root data/abo_500 \
  --mesh-root data/abo_500_meshes \
  --mesh-unit-to-meter 0.001 \
  --config configs/abo500_mesh.yaml \
  --split test \
  --limit 2 \
  --output outputs/debug_all_solid
```

成功后运行完整 train/val/test。校准器只在 train 拟合：

```bash
bash scripts/run_abo500_mesh.sh \
  data/abo_500 \
  data/abo_500_meshes \
  0.001 \
  outputs/abo500_all_solid
```

检查：

```text
outputs/abo500_all_solid/mass_calibration.json
outputs/abo500_all_solid/test/metrics.json
outputs/abo500_all_solid/test/run_summary.json
```

`run_summary.json` 中的失败数、水密失败和实际成功样本数必须写入实验记录。

## 5. 结构化内部 VLM

先启动一个与 OpenAI API 兼容的多模态模型服务，并设置：

```bash
export TACTILE3D_VLM_BASE_URL=http://127.0.0.1:8000/v1
export TACTILE3D_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
```

先运行两个样本：

```bash
bash scripts/run_abo500_mesh_vlm.sh \
  data/abo_500 \
  data/abo_500_meshes \
  0.001 \
  outputs/abo500_vlm_debug \
  2
```

检查每个场景的：

```text
interior_proposals.json
prediction.json
part_crops/
```

确认 JSON 合法、部件裁剪确实对应目标部件、结构概率归一化后，再去掉最后的 `2` 运行完整
实验。将响应缓存后可使用 `--vlm-backend file` 重复物理积分，避免每次重新调用 VLM。

## 6. QLoRA 微调

专家标注应遵循 `docs/expert_annotation.example.jsonl`，并按物体 ID 分组切分，防止同一物体
不同视角跨 train/test：

```bash
tactile3d prepare-sft \
  --annotations data/annotations/expert_parts.jsonl \
  --image-root data/renders \
  --max-images 4 \
  --output data/sft
```

两张 4090 运行：

```bash
bash scripts/train_vlm_2x4090.sh \
  data/sft \
  outputs/qwen3vl_8b_qlora
```

默认参数为每卡 batch size 1、梯度累积 8、LoRA rank 32、3 epochs。显存不足时先把每条样本
视图数从 4 降至 2，再降低 `--max-pixels`，不要先改测试协议。

当前训练输出是 LoRA adapter。正式用于 VLM 推理前，需要由推理服务加载该 adapter 或先与
基础模型合并，并记录基础模型 revision、adapter checkpoint 和服务版本。仓库目前没有自动
部署 adapter 的脚本，这是正式微调复现前需要补齐的一项工程工作。

## 7. 正式消融矩阵

所有方法固定 seed、split、mesh 和尺度：

```text
A0  geometry + uniform density
A1  OpenCLIP surface + all-solid
A2  OpenCLIP surface + heuristic interior
A3  OpenCLIP surface + base VLM interior
A4  OpenCLIP surface + fine-tuned VLM interior
A5  A4 + train-only mass calibration (full model)
```

至少运行三个随机种子，例如 17、29、43。每个方法只允许在 validation 选择设置，test 只运行
最终配置。若某方法失败，记录失败率，不能只在该方法中删除样本。

## 8. 基线与论文指标

将 PUGS/NeRF2Physics 输出转换成统一格式：

```bash
tactile3d import-mass-baseline \
  --predictions results/pugs_predictions.json \
  --targets data/abo_500/filtered_product_weights.json \
  --method PUGS \
  --output outputs/baselines/pugs.json
```

生成共同测试对象上的 ADE、ALDE、APE、MnRE、bootstrap 置信区间和不确定性图：

```bash
tactile3d paper-figures \
  --run NeRF2Physics=outputs/baselines/nerf2physics.json \
  --run PUGS=outputs/baselines/pugs.json \
  --run PartPhy3D=outputs/abo500_full/test/predictions.json \
  --cohort configs/abo500_mass_test.json \
  --evaluation-context published_test \
  --bootstrap-samples 2000 \
  --seed 17 \
  --output outputs/paper/comparison
```

核心文件：

```text
comparison_metrics.json
comparison_source_data.csv
mass_metrics_comparison.{svg,pdf,png,tiff}
predicted_vs_ground_truth.{svg,pdf,png,tiff}
per_object_ape.{svg,pdf,png,tiff}
uncertainty_calibration.{svg,pdf,png,tiff}
```

## 9. 单物体效果图

```bash
tactile3d object-figures \
  --prediction outputs/abo500_full/test/scenes/SCENE_ID/prediction.json \
  --mesh outputs/abo500_full/test/scenes/SCENE_ID/mesh_metric.ply \
  --face-labels outputs/abo500_full/test/scenes/SCENE_ID/face_labels.npy \
  --output outputs/paper/objects/SCENE_ID
```

该命令生成部件分割、有效密度、有效杨氏模量、部件质量区间和内部结构概率图。ABO-500 没有
部件级物理真值，因此这些图是定性推断，不应写成定量 ground truth。

## 10. 判定是否完成复现

只有同时满足以下条件，才能称为复现了当前方案：

- 测试与 smoke test 全部通过。
- ABO-500 split、scene ID、相机与 mesh 坐标一致。
- 每个 mesh 的真实尺度有来源，水密失败被显式记录。
- 校准器只使用 train，模型选择只使用 validation。
- test 上报告 ADE、ALDE、APE、MnRE、样本数和失败率。
- 概率模型同时报告 90% coverage 与区间宽度。
- PUGS、NeRF2Physics 和本方法在同一对象交集上比较。
- 保存 Git commit、依赖、GPU、模型 revision、随机种子、配置和 source data。
- 明确区分真实 ABO-500 结果、合成 smoke 结果和无真值的定性部件图。
