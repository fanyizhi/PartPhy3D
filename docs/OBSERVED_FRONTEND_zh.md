# 六张真实标定图像的冻结重建器诊断

此入口用InstantMesh现成的重建网络验证图像、相机、生成Mesh和评价数据流。
它不运行Zero123，不是官方单图流程，也不是PartPhy3D提出的新生成网络。
参考GLB和质量标签均不参与重建。当前完整方法尚未实现。

## 安装和资产

Python 3.10、RTX 4090 24 GB是本轮实测配置。统一环境同时运行Qwen和该重建
适配器，不需要为此降级主环境Transformers。项目根目录中执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_installation.py --training --frontend
```

requirements包含固定版本的einops、OmegaConf、xatlas、OpenCV和经SHA256
校验的nvdiffrast上游包。安装需要访问相应软件源，包括codeload.github.com。
该入口只导出顶点色几何，不调用可微光栅器，所以不需要为它单独编译CUDA插件。
其他使用nvdiffrast可微渲染的程序仍可能需要匹配的CUDA编译工具链。

准备冻结上游代码；不要用未记录的最新版替代：

```bash
mkdir -p external
git clone https://github.com/TencentARC/InstantMesh.git external/InstantMesh
git -C external/InstantMesh checkout 08822c52fdc399b93ea00e4fa9e596344ed52ccc
```

只下载实际使用的重建检查点；清单固定文件版本、大小和SHA256：

```bash
mkdir -p weights/InstantMesh-observed
cp configs/pretrained/instantmesh_checkpoint_manifest.json \
  weights/InstantMesh-observed/download_manifest.json
python scripts/download_model_snapshot.py --model TencentARC/InstantMesh \
  --output weights/InstantMesh-observed
```

需要约1.5 GB的`instant_mesh_large.ckpt`。无需下载其他InstantMesh检查点、
Zero123扩散权重或单独的DINO权重。DINO的架构和图像处理配置已固定于
`configs/pretrained/dino-vitb16/`，来源与哈希在SOURCES.json中。完整InstantMesh
检查点必须严格加载全部参数，包含DINO；缺少权重时失败，不以随机参数继续。
第三方代码、模型和数据遵循各自许可，见[第三方说明](../THIRD_PARTY.md)。

## 单对象重建与独立视角检查

先处理冻结队列的第一个物体：

```bash
python scripts/run_observed_instantmesh.py \
  --scene data/abo500_five/scenes/B07B4G5RBN_ATVPDKIKX0DER \
  --repository external/InstantMesh \
  --checkpoint weights/InstantMesh-observed/instant_mesh_large.ckpt \
  --encoder-config configs/pretrained/dino-vitb16 \
  --output outputs/observed_first
python scripts/evaluate_observed_mesh.py \
  --run outputs/observed_first --output outputs/observed_first_evaluation
```

目录含原始输入哈希、相机归一化、实际选中视图、代码哈希、生成Mesh和运行报告。
失败保留failure.json和traceback，不覆盖原实验。评价采用同一标定下的30个视角，
分别报告6个输入视角和24个未输入视角的轮廓IoU；不拟合参考Mesh或搜索最优对齐。

完整五物体逐个执行，控制GPU峰值；不要同时启动五个进程：

```bash
for id in B07B4G5RBN_ATVPDKIKX0DER B076YKYHCB_A1F83G8C2ARO7P \
          B07WTFJGRG B073WGP8XN B013TGFGWS_A1F83G8C2ARO7P; do
  python scripts/run_observed_instantmesh.py \
    --scene "data/abo500_five/scenes/$id" \
    --repository external/InstantMesh \
    --checkpoint weights/InstantMesh-observed/instant_mesh_large.ckpt \
    --encoder-config configs/pretrained/dino-vitb16 \
    --output "outputs/observed_five/$id" || break
  python scripts/evaluate_observed_mesh.py \
    --run "outputs/observed_five/$id" \
    --output "outputs/observed_five/${id}_evaluation" || break
done
```

## 尺度与方法边界

相机采用无畸变的OpenGL c2w、Z向上约定。由相机视轴求公共中心，再将平均相机
半径归一化为4。相似变换保持投影，单元测试核对该性质并拒绝退化标定。
六个输入视角按最接近固定方向的规则无放回选取，不看物体类别、质量或评价分数。

`camera_space.ply`还原到输入相机单位；这不等于独立获得米制尺度。
`camera_unit_to_meter=null`时不输出m³体积，也不把它与kg/m³相乘当成真实质量。
水密仅描述三角网格闭合，不能确认空腔、泡棉、骨架、壁厚或材料分区。

冻结网络内部采用规则网格上的FlexiCubes提取；本轮没有执行我们的自适应生成
预算、精细语义部件模型或物理约束学习。后续应在相同输入/尺度和独立标签下
验证这些改进，避免把接通上游代码当成创新效果。

[本轮实际结果、失败和下一阶段任务](results/server_pilot_20261106/README_zh.md)。
