# ABO-500真实质量监督训练

三轮训练与前后测试已完成，见[实际结果与失败分析](results/abo_mass_sft_20261107/README_zh.md)。

此实验训练的是图像到质量的VLM基线。它使用ABO质量元数据，不包含未经核验的
内部材料、密度、杨氏模量或部件掩码答案，也不是完整的3D物理联合训练。

使用官方300/100/100对象划分，按ASIN检查跨划分重叠，并拒绝完全重复的图像。
每个物体固定使用第00张图片，透明通道合成白底，派生为384×384 PNG。
转换图像有原图SHA256、派生图SHA256和固定prompt记录。测试输入文件不含答案。
官方质量表按ASIN索引，带市场后缀的渲染场景须显式映射，不能直接按场景名索引。

## 完整运行

主环境采用统一requirements；需要准备完整ABO-500目录和已校验的本地8B权重。
以下命令在Linux项目根目录执行；输出目录须不存在。

```bash
python scripts/prepare_abo_mass_sft.py \
  --root /path/to/abo_500 \
  --output data/abo500_mass_sft \
  --runtime-root "$PWD/data/abo500_mass_sft"

python scripts/evaluate_mass_sft.py \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --dataset data/abo500_mass_sft --output outputs/mass_run/before

tactile3d train-vlm --model /path/to/Qwen3-VL-8B-Instruct \
  --train-data data/abo500_mass_sft/train.jsonl \
  --val-data data/abo500_mass_sft/val.jsonl \
  --output outputs/mass_run/adapter --qlora \
  --epochs 3 --learning-rate 5e-5 --batch-size 1 \
  --gradient-accumulation 4 --eval-steps 25 --logging-steps 5 \
  --lora-rank 8 --lora-alpha 16 --min-pixels 12544 --max-pixels 147456

python scripts/evaluate_mass_sft.py \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --dataset data/abo500_mass_sft --output outputs/mass_run/after \
  --adapter outputs/mass_run/adapter
```

每次命令返回成功后再运行下一条。QLoRA为NF4双重量化、BF16计算、只监督
assistant答案。随机种子与数据种子均为17。检查点最多保留两份，最终适配器
独立保存，不覆盖基座。300样本、batch=1、累积4、3轮对应225次更新。

## 如何判断训练有效

训练前后采用相同NF4量化、同一提示、相同384×384图像及贪心解码，避免将量化
或输入分辨率变化误算成训练收益。评价代码保存原始回答和失败，再封存预测，
最后读取测试质量标签。必须同时报告有效输出覆盖率、ADE、ALDE、APE及RMSE。

验证损失下降只表明答案似然提高，不能代替生成质量数值的准确率。该轮按固定
三轮训练，不根据测试误差选择轮数。首批5个测试对象已经用于近期开发，历史规则先导还使用过前10个。应同时报告
全100对象、排除近期5例的95对象，以及排除历史10例的90对象；这些诊断不能
直接替代独立采集的盲测。

ASIN划分无法自动排除外观相近的产品族，也无法排除基座预训练见过公开ABO资产。
质量元数据可能有噪声，不等于我们独立称重。一次初始化只能作为先导结果；后续
应增加重复种子、专业数据与独立物理观测，验证收益是否稳定。
