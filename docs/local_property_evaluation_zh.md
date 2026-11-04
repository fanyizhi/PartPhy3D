# 局部物理属性的定量比较

与总质量评价并列报告局部物理属性准确性。预测与参考必须对应同一个物体、同一位置或部件、同一种物理含义和测量条件。比如部件外包络的有效密度不能直接对比木材本征密度；夹爪接触刚度也不能直接当作杨氏模量。

## 指标与真值

| 属性 | 规范单位 | 主要指标 | 所需独立参考 |
| --- | --- | --- | --- |
| 密度rho | kg/m^3 | MAE、RMSE；正值时ALDE、APE、MnRE | 材料样本质量/体积，或结构已知的空间密度 |
| 杨氏模量E | Pa | ALDE、MAE/RMSE及相对误差 | 适用载荷/边界/方向下的力学测量或明确的合成参考 |
| 泊松比nu | 1 | MAE、RMSE | 拉伸等实验或已知材料参考；负泊松比可有效，勿取对数 |
| 摩擦系数mu | 1 | MAE、RMSE；正值时ALDE、APE、MnRE | 指定材料对、静/动状态、载荷与环境下的摩擦测量 |
| Shore A/D硬度 | 对应标尺 | 同标尺MAE/RMSE，正值比值指标需说明解释 | 对应硬度计实测 |
| 接触刚度K | N/m | MAE、RMSE，力—位移曲线误差另报 | 指定压头、速度、支承和载荷范围下的测量 |

MAE与质量评价中的平均绝对差同型，单位随属性变化。ALDE=mean(abs(ln(pred)-ln(target)))；APE=mean(abs(pred-target)/target)；MnRE=mean(min(pred/target,target/pred))。默认先按物体汇总各采样点误差，再对物体等权平均，避免采样点多的物体支配结果。RMSE定义为宏平均MSE的平方根。

零值处的相对/对数指标可能无定义，评价器保留绝对误差，明确报告各指标有效点数；不静默加epsilon。泊松比只报告绝对指标。缺失参考不填0、不产生分数；有参考但缺失预测时报告覆盖率和不完整状态。

## 与论文协议对应

- NeRF2Physics在真实场景上测量了逐点动摩擦与Shore硬度，并报告误差与相对关系准确性PRA。原文为比较使用A读数0–100、D读数加100的拼接编码；这是特定评价编码，不是A/D物理换算。[原文第4.3节](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhai_Physical_Property_Understanding_from_Language-Embedded_Feature_Fields_CVPR_2024_paper.pdf)。
- PhysVGGT表2同样报告NeRF2Physics真实场景上的摩擦和硬度指标，其训练局部属性包含自动生成的伪标签。[原文](https://arxiv.org/html/2609.18920v1)。
- VoMP展示了与基线的E、nu、rho空间字段对比；用于数值评价时还须采用其相应数据与参考协议。[作者项目页](https://research.nvidia.com/labs/sil/projects/vomp/)。

正式主表可分为：质量；摩擦/硬度实测基准；E/nu/rho已知构造或力学参考基准。实测、已知合成与伪标签结果分组，不相互冒充。相同色标属性图用来观察空间分布，配套真值图与误差图才支持准确性判断；缺少真值时标注为定性结果。

## 已实现的评价接口

入口为`src/tactile3d/eval/properties.py`，输入长表CSV：

```text
method,object_id,sample_id,property,unit,protocol,predicted,target,reference_kind,reference_source
```

`sample_id`标识同一测量点或部件；`protocol`声明测量协议和量的含义。`reference_kind`允许`measured`、`known_synthetic`或`pseudo_label`，分别评分。单位必须预先明确转换到规范单位，不能把Pa与GPa直接混合。不同方法对比时额外核对有效样本ID集合，不能只比较不同子集的均值。

本次贵妃椅已生成：

- `outputs/abo500_single_object_v2/local_property_reference_template.csv`：11个部件×7种属性；只填写已有的部件有效密度预测，其他预测与全部测量参考保持空白。
- `outputs/abo500_single_object_v2/local_property_evaluation.json`：目前7个属性组全部为`reference_unavailable`，没有伪造局部准确性分数。E混合界没有当成点预测填入。

取得独立参考且完成对应预测后，保存到一个新的CSV，再运行：

```bash
python -m tactile3d.eval.properties \
  --input outputs/local_properties_measured.csv \
  --output outputs/local_property_metrics.json
```

已通过5项针对性检查：物体等权汇总、缺失参考/零值处理、负泊松比、参考来源分组、单位/重复值/非有限值检查及缺失预测覆盖率。

当前贵妃椅的质量标签接近预测，并不能验证其内部密度、杨氏模量、摩擦或硬度准确。下一阶段应先接公开实测摩擦/硬度数据，再补有明确构造和力学参考的E/nu/rho评价；不以材料查表或模型自己的推断充当独立真值。
