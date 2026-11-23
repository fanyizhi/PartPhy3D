# YCB 实物扫描锤的 MuJoCo 抛投

原输入为 YCB 048_hammer Google 16k 的闭合扫描网格，CC BY 4.0。长轴用 PCA 确定，较宽一端对应锤头。将长轴从柄底起 80% 处作为演示切分面。体积用 2 mm 体素计算组分比例，再令总和等于原闭合网格体积。该分割边界和锤柄等效密度属于本算例假设。钢头取 7,850 kg/m³，铝头取 2,700 kg/m³，手柄取等效密度 1,050 kg/m³。编辑只改锤头参数。均匀组使用钢头组同样的总质量。

三组用同一物体坐标系起抛，使用相同线冲量和角冲量。重力仅施加于各自质量中心。统一密度和钢头组质量相同，因而质心从各自初始点算起的位移曲线重合。两组初始质心位置、惯量与同一锤头标记的轨迹不同。铝头组质量改变，等冲量下初速度也改变。上排是跟随各组质心的姿态视图，对照用同一相机投影与视图尺度。下排保持世界坐标的水平与竖直量。圆点为计算质心，十字为附着于扫描锤头的同一个标记。运动状态由 MuJoCo 求解，网格与曲线由 Matplotlib 合成。

钢头组预测 0.893 kg，YCB 原文表 1 实测为 0.665 kg，偏高 34.35%。原始填充体素体积比网格闭合体积高约 20.2%，已做总体积归一化，组分比例仍可能受采样或切分影响。我们保留这一误差。数据没有提供这把锤的实测质心和同条件抛投真值，当前结论限于局部改材与空间分配的响应变化。实验参考 Gen2Physics 图 1 用同一几何比较质量分布的思想，使用的是 YCB 锤和我们自己的起抛协议。

[YCB 资产](https://ycb-benchmarks.s3.amazonaws.com/index.html) · [YCB 实测质量表](https://www.eng.yale.edu/grablab/pubs/Calli_IJRR2017.pdf) · [Gen2Physics 图 1](https://arxiv.org/html/2608.23869v1#S1) · [视频](ycb_hammer.mp4) · [论文图](ycb_hammer.pdf) · [逐帧记录](tracks.npz)

论文补充视频为 Blender 上排渲染与世界坐标轨迹图的合成。上排镜头跟随各组质心以放大可读性。下排保持世界坐标比例和单位。使用原扫描的三角面几何，材质颜色为实验标注。金色点指同一个头部固定标记，彩色点指各组估计质心。[同步曲线视频](ycb_hammer_research.mp4) · [论文静态图](ycb_hammer_research.pdf) · [参考排版的完整飞行轨迹](flight_overlay/README.md)

复现时先运行 `scripts/run_ycb_hammer_projectile.py` 生成轨迹及渲染几何数据。Blender 4.3.2 运行 `scripts/render_ycb_hammer_closeups.py`，并用 `scripts/compose_ycb_hammer_research_video.py` 合成。全程轨迹使用求解脚本的 `--full-flight` 选项，再调用 `scripts/render_ycb_flight_overlay.py`。
