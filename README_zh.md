# PartPhy3D

面向部件构造、物理属性推断和物理一致编辑的研究工程，当前版本 **0.5.0**。
源码包含可执行程序和研究设计；尚未完成真实联合模型，不能据此宣称优于论文方法。

[English](README.md) · [完整方法、程序与复现](docs/METHODS_AND_REPRODUCTION_zh.md) ·
[实现状态](docs/IMPLEMENTATION_STATUS.md) · [运行结果和不足](docs/results/reproduction_v0.5.0_zh.md)

## Linux完整CPU验收

在MobaXterm终端中：

```bash
git clone https://github.com/fanyizhi/PartPhy3D.git
cd PartPhy3D
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,figures]"
python -m pytest -q
python scripts/reproduce_all.py --output outputs/first_run
```

建议先用Python 3.10；Linux CI检查3.10，本地另测3.13。
CPU验收不需要4090、torch、模型权重或ABO，也不连接服务器或恢复队列。
请在新环境安装，不要直接升级其他人正在使用的共享环境。
完整依赖统一在`requirements.txt`，不再分`requirements-train`。
CPU安装跳过大模型包；完整环境和模型准备见详细手册。

## 一条命令实际完成什么

1. 用独立解析构造生成384个合成对象，按对象族划分训练、校准、测试集。
2. 用三种初始化训练质量监督、共享状态＋编辑响应、直接回归三个对照。
3. 评测前封存预测，输出原物体、已见材料编辑、未见钢材编辑的质量误差。
4. 输出构造分类、密度、宏观占用和体积误差，以及混合预测和MAP资产质量差距。
5. 导出预测的可编辑状态，改为钢材后重新积分质量、质心和完整惯量。
6. 验证独立解析装配体的材料、空心构造、部件替换、接口检查和事务重放。
7. 对比均匀与任务加权曲面构造，记录面数、场查询数、误差、水密性和耗时。
8. 生成PNG/SVG图、CSV/JSON、NPZ、PLY和SHA-256产物清单。

**学习实验使用11维观测代理特征，不是图像编码器或千问微调，也不是ABO结果。**
真实模型入口和数学验收链分别记录，不把占位模块称为训练成功。

## 图片和数据在哪里

```text
outputs/first_run/learning/learning_results.png       质量散点、误差和校准区间
outputs/first_run/learning/physical_fields.png        密度、占用及材料编辑切片
outputs/first_run/learning/metrics.csv                原物体和编辑质量指标
outputs/first_run/learning/state_metrics.csv          密度、构造、占用指标
outputs/first_run/learning/seed_17/example/           原始/编辑状态和PLY
outputs/first_run/editing/                           三种编辑及物理、几何变化图
outputs/first_run/adaptive/adaptive_construction.png 稀疏构造成本与误差
outputs/first_run/suite.json                         整体状态、作用范围和耗时
```

图同时导出SVG，CSV提供数值来源。目录存在时拒绝覆盖，请换新目录。
程序不会隐藏失败，也不会把未运行的论文分数补成零。

## 真实数据与模型

- ABO图像、相机和称重标签放在`data/abo_500/`，参考GLB与生成Mesh分目录。
- 已下载资产支持`data/abo500_ten/<scene_id>/model.glb`。
- `scripts/reconstruct_instantmesh.py`适配官方`run.py`，默认仅检查；显式增加
  `--execute`才运行。由单张观测生成的辅助视角不能当作真实多视角。
- `configs/abo500_mesh_vlm.yaml`连接已部署的千问兼容服务；缺失依赖时明确失败。
- 几何聚类是当前基线，不等于PartField或精细语义分割。
- 非水密Mesh可明确选`voxel_envelope`近似包络，不等于恢复真实空腔和材料。
- 真实图像→精细部件→学习构造状态→编辑的联合模型仍待完成和独立验证。

两物体检查命令、文件格式和实际误差见[完整手册](docs/METHODS_AND_REPRODUCTION_zh.md)。
两个物体的数据通路实测成功，但规则质量预测很差，失败与误差已如实记录。

## 打包与安装

```bash
python scripts/package_project.py --output outputs/release --results outputs/first_run
```

输出源码ZIP、wheel、sdist、CPU结果ZIP和校验文件。
不含ABO、模型权重、私钥或服务器配置，第三方依赖仍需通过pip安装。
拿到wheel后可在其他环境安装运行：

```bash
python -m pip install "outputs/release/dist/tactile3d-0.5.0-py3-none-any.whl[figures]"
partphy3d-reproduce --output outputs/wheel_check
```

## 研究说明

- [每一步的方法、代码、损失、指标和实验](docs/METHODS_AND_REPRODUCTION_zh.md)
- [部件状态、版本、接口和编辑契约](docs/part_state_and_edit_contract_zh.md)
- [编辑指令与重放](docs/edit_commands_2026-11-04_zh.md)
- [完整论文设计和验收条件](docs/research_blueprint_2026-11-04_zh.md)
- [问题、方法和待验证优势](docs/problem_solution_advantages_2026-11-04_zh.md)
- [公平对比协议](docs/evaluation_protocol_zh.md)
- [上游方法与许可](THIRD_PARTY.md)

统一数据结构、标准损失和红绿细分不自动构成原创贡献。需要公平对照证明：
共享可执行构造状态与独立编辑响应监督能否提高**原物体及编辑后的物理准确性**，
并在几何和物理误差约束下减少资源开销。
