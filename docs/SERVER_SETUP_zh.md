# 一次下载、统一安装、服务器校验

固定五物体的元数据位于`data/abo500_five/`，图片和参考模型另行在本地准备。
项目根目录中的
`requirements.txt`一次安装几何处理、绘图、测试、视觉与微调依赖，以及项目本身。
`environment.yml`只负责创建Python和pip，并调用同一份requirements。

## 通过GitHub下载

建议使用Python 3.10，在MobaXterm服务器终端：

```bash
git clone --depth 1 https://github.com/fanyizhi/PartPhy3D.git
cd PartPhy3D
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_installation.py --training --skip-data
```

仓库仅包含源码、最新结果和数据元信息。按[数据准备](DATA.md)恢复150张图和参考模型后，
再运行不带`--skip-data`的检查。已有本地或服务器数据无需重新下载。

## 使用带数据的完整ZIP

在已准备好数据的本机项目根目录可生成迁移包：

```bash
python scripts/package_server.py --output outputs/server_bundle
```

把`PartPhy3D-with-data.zip`及其`.sha256`文件放到服务器同一目录，然后：

```bash
sha256sum -c PartPhy3D-with-data.zip.sha256
unzip PartPhy3D-with-data.zip
cd PartPhy3D
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_installation.py --training
```

依赖由pip联网下载；ZIP内含源码、配置、文档和数据，没有打包各操作系统的
第三方wheel、CUDA驱动和预训练权重。相应模型权重准备见原方法手册。

## 安装版本与检查范围

耦合依赖固定为PyTorch 2.6.0、TorchVision 0.21.0、Transformers 4.57.6、
TRL 0.29.0、PEFT 0.17.1及Datasets 3.6.0。详细版本以requirements为准。
保留兼容CPU/CUDA构建的版本规格；服务器实际CUDA构建及驱动需连接后核对。
先使用本项目的新环境，不修改共享训练环境。

完整安装的Linux CI使用CPU版PyTorch验证全部依赖和训练API的导入：

```bash
python -m pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

CPU验证没有执行GPU训练。`check_installation.py`检查实际导入、`pip check`及
五物体数据哈希；`--training`额外检查训练库和需要的类，不下载权重、不执行推理、
不分配GPU张量。返回`passed`后，可以先执行软件验收：

```bash
python -m pytest -q
python scripts/reproduce_mesh_native.py --output outputs/mesh_first
```

上面的Mesh复现是独立的解析/合成诊断。真实五物体程序默认读取内置数据：
`python scripts/run_five_objects.py --help`。正式运行仍需要核验尺度、相机配准和
选定模型，不能把参考Mesh复制成功当作生成和物理预测成功。

## 数据来源

ABO资产使用[CC BY 4.0](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)，
ABO-500观测和标签来自[NeRF2Physics发布](https://github.com/ajzhai/NeRF2Physics)。
子集附带`ATTRIBUTION.md`与原许可文本。五物体是固定开发队列，不是独立最终测试集。
缓存的上游材料建议仍为建议，不是新增实测内部材料真值。
