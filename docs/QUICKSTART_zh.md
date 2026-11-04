# 安装、数据与复现

## 无数据CPU验证

在项目根目录使用Python 3.10及以上：

```bash
python -m pip install -e ".[dev,figures]"
python scripts/edit_parts.py
python scripts/edit_parts.py --plan examples/wood_to_hollow_steel.json
python -m pytest -q
```

可编辑安装避免`tactile3d`导入路径错误。本机实际执行环境是Windows/Python 3.13；
配置的Linux CI需要推送后实际运行。无需为该演示安装torch或下载模型。

输出到新的`outputs/edit_commands/run_<编号>`。默认指定设计的总质量依次为
0.7840、3.0880、2.5228、0.8568 kg，不是ABO质量预测。完成的`manifest.json`
及对应哈希用于确认导出完整性。文件写入失败时保留错误，并使用新目录重试。

## 完整研究环境

完整依赖统一保留在`requirements.txt`，conda路线保留在`environment.yml`。
训练依赖不是CPU快速验证的前置条件。torch/CUDA/模型兼容性需在目标环境实测，
本机测试不能证明服务器训练成功。

## 数据布局

```text
data/abo_500/splits.json
data/abo_500/filtered_product_weights.json
data/abo_500/scenes/<object_id>/
data/abo500_ten/<object_id>/model.glb
data/material_color_demo/source.png
data/material_color_demo/source.json
data/material_color_demo/camera.json
```

ABO-500与原生GLB分别取得并遵守原许可；无face的PLY是点云。参考Mesh诊断与
生成Mesh主流程分开。测试质量不得进入提示词、参数优化或几何选择。

颜色预览相机JSON包含`w,h,fl_x,fl_y,cx,cy,camera_to_world`，矩阵对应所选原始
视角。准备数据后运行`python scripts/preview_material_edit.py`。该入口目前只改
参考Mesh的选定表面颜色，没有提交内部物理状态。

## 原物体比较与发布

已有并封存十物体预测目录时：

```bash
python scripts/compare_experiments.py --pilot outputs/abo500_ten_v2
```

源码包不附带该中间目录。汇总记录在`docs/results/`。没有运行的方法不填0，
不以论文均分替代同对象预测。先统一输入、尺度、监督和失败统计再计算指标。

构建源码发布包：

```bash
python scripts/build_release.py
```

发布包含源码、配置、示例、测试、文档和CI，不含数据、权重、运行日志、缓存
及本机SSH诊断脚本。哈希清单只验证包完整性，不证明科学结论。
