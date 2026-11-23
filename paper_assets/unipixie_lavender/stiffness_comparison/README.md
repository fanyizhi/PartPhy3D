# 紫花受同一阵风的刚度对照

对象为 Eryn_Noel 的 Assignment4_Flowers。原 GLB 在 PixieVerse 索引中对应 Objaverse UID 18f5fba66f944aad9ae84ea2172b9b78，作者预览与 UniPixie 图 4A 外观吻合。原论文演示把柔性植物运动过硬与过高的杨氏模量联系起来。论文使用模拟参考轨迹作评估，报告的物性指标还包括密度与泊松比。

这里比较两个指定模量 90 MPa 和 360 MPa。整株 0.45 m，等效茎半径 2.5 mm，密度 650 kg/m³，阻尼比 0.15。两组使用同一原纹理 GLB、固定根部和同一峰值 1 m/s 阵风。对每个网格节点用悬臂梁等效长度，分别利用 EI 给出静态柔度和首阶频率。在 MuJoCo 中，每个节点用一个带弹簧与阻尼的平移自由度表示尖端响应。用同一梁形函数把求得的位移映射回原网格。画面按 1 倍位移显示。独立求解相同连续方程来校验 MuJoCo 的积分。

这是刚度敏感性对照。模量、尺度、阻尼和截面是条件设定。求解采用降阶梁模型，同行对照和真实材料准确性需要相同边界与载荷下的逐帧参考及基线运动。

[视频](lavender_wind.mp4) · [静态论文图](lavender_wind.pdf) · [全部曲线](curves.npz) · [参数和残差](result.json) · [UniPixie 图 4](https://arxiv.org/html/2606.05399v1#S4.SS2) · [原模型及署名](https://sketchfab.com/3d-models/18f5fba66f944aad9ae84ea2172b9b78)

两档模量的峰值分别为 13.58 mm（0.96 s）和 2.78 mm（0.84 s）。最长节点的无阻尼首阶频率分别为 1.347 和 2.694 Hz。十个降阶模型在采样时刻相对独立常微分方程的最大位移误差低于 2.0×10⁻⁸ m。

UniPixie 论文 §3.4 采用 PhysGaussian 的 MPM 路径，将带物性的 3D Gaussians 作为材料点。PhysGaussian 的公开运行脚本使用 NVIDIA Warp 做 MPM、Gaussian rasterizer 逐帧渲染、FFmpeg 编码。前代 Pixie 的公开项目使用 Blender 4.3.2 做资产渲染。UniPixie 项目页没有公开图 4 案例的剪辑工程或逐实例渲染设置。[UniPixie §3.4](https://arxiv.org/html/2606.05399v1#S3.SS4) · [PhysGaussian 实现](https://github.com/XPandora/PhysGaussian/blob/main/gs_simulation.py) · [Pixie 公共代码](https://github.com/vlongle/pixie)

Blender 4.3.2 的 Cycles 渲染以两组 MuJoCo 位移轨迹驱动原 GLB 网格的形状键，使用固定正交摄影机和相同布光。该视频与论文曲线图共用同一组求解数据。可用脚本 `scripts/render_lavender_blender.py` 从原资产和本目录重建已打包纹理的 Blender 工程。对服务器补充了缺失的 libSM 和 libICE。打包操作使用 Blender 4.3.2 的 file.pack_all。 [Blender 输出](lavender_blender.mp4)
