# PartPhy3D 服务器测试

## 1. 当前已实现功能

| 功能 | 状态 | 主要输出 |
|---|---|---|
| 水密 mesh 修复、检查和精确体积 | 已实现 | `volume_report.json` |
| 几何部件分割与内部体素分配 | baseline 已实现 | `mesh_parts.ply`, `face_labels.npy` |
| 多视角部件投影与材料证据融合 | 已实现 | 部件 crops、材料概率 |
| VLM 表面/内部结构候选 | 接口已实现 | `interior_proposals.json` |
| 实心、壳体、泡沫、框架、填充质量积分 | 已实现 | `prediction.json` |
| Monte Carlo 不确定性 | 已实现 | mean/std/q05/q50/q95 |
| ABO-500 train-only 质量校准 | 已实现 | `mass_calibration.json` |
| ADE/ALDE/APE/MnRE 与论文图 | 已实现 | metrics、CSV、SVG/PDF/TIFF |
| Qwen3-VL LoRA/QLoRA 训练入口 | 已实现 | LoRA adapter |
| donor 部件替换、ICP 和接口检查 | baseline 已实现 | `edited_assembly.ply` |
| 编辑后 Mass/COM/Inertia 重构 | 已实现 | `edit_report.json` |
| 静态支撑与刚体属性解析检查 | 已实现 | consistency report |

尚未完成：RGB 自动生成 mesh、学习式语义部件分割、统一 `PartPhysicalState/edit_spec`、编辑后
分层 occupancy 重新体素化、自动 geometry adaptation gate、真实刚体/软体模拟闭环。

## 2. 上传与安装

将本机项目目录同步至服务器，例如：

```bash
rsync -av --exclude outputs/ tactile3D/ USER@SERVER:~/PartPhy3D/
ssh USER@SERVER
cd ~/PartPhy3D
```

安装：

```bash
conda env create -f environment.yml
conda activate partphy3d
```

## 3. 第一档：不需要真实数据

```bash
bash scripts/server_smoke_test.sh
```

该脚本检查 GPU/Blender 信息、运行全部测试、创建五个合成物体、执行训练集质量校准、测试集
评测、水密体积计算和部件替换，并验证编辑后的质量、质心和惯量一致性。

通过标志：最后输出 `PASS: PartPhy3D server smoke test completed`。

## 4. 第二档：两个真实 ABO 对象

假设原始 GLB 已经正确匹配并整理到 `data/abo_500_meshes/<scene_id>.glb`：

```bash
bash scripts/server_smoke_test.sh \
  data/abo_500 \
  data/abo_500_meshes \
  0.001
```

`0.001` 只适用于一 mesh unit 等于一毫米的情况。原始 GLB 若采用其他单位，必须使用经过
商品尺寸或元数据验证的值。该阶段将首次下载 OpenCLIP 权重，需要服务器可以访问模型文件。

检查：

```text
outputs/server_smoke/abo500_real_two/run_summary.json
outputs/server_smoke/abo500_real_two/predictions.json
outputs/server_smoke/abo500_real_two/metrics.json
```

若 `num_failed > 0`，先检查 scene/GLB 映射、相机坐标、纹理、mesh 水密性和尺度，不要立即
修改密度或校准参数来掩盖数据问题。

## 5. 第三档：VLM 两对象测试

启动 OpenAI-compatible 多模态服务后：

```bash
export TACTILE3D_VLM_BASE_URL=http://127.0.0.1:8000/v1
export TACTILE3D_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct

bash scripts/run_abo500_mesh_vlm.sh \
  data/abo_500 \
  data/abo_500_meshes \
  0.001 \
  outputs/abo500_vlm_debug \
  2
```

逐个查看 `part_crops/` 和 `interior_proposals.json`。只有裁剪对应正确部件、JSON 合法、结构
概率和材料体积分数合理后，才能运行完整 500 对象。

## 6. 两张 4090 的 QLoRA 测试

准备真实专家标注后：

```bash
tactile3d prepare-sft \
  --annotations data/annotations/expert_parts.jsonl \
  --image-root data/renders \
  --max-images 4 \
  --output data/sft

bash scripts/train_vlm_2x4090.sh data/sft outputs/qwen3vl_8b_qlora
```

先运行少量训练数据并确认两张卡均有显存占用、loss 有限且 validation 没有数据泄漏，再运行
三轮正式训练。
