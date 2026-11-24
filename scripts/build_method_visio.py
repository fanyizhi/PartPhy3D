"""Create the editable Chinese method walkthrough with Microsoft Visio on Windows."""

from pathlib import Path
import win32com.client as com
import pythoncom

OUT = Path(__file__).resolve().parents[1] / "paper_assets" / "method_overview"
INK = "#183348"
COLORS = {
    "input": ("#F1F4F6", "#B8C7D0"),
    "geometry": ("#EBF4FA", "#8CB7CC"),
    "material": ("#E6F3EE", "#84BAAA"),
    "physics": ("#FBF0E3", "#D7AB7A"),
    "result": ("#EDEBF6", "#A8A0C4"),
}


def rgb(value):
    return "RGB({},{},{})".format(*(int(value[n : n + 2], 16) for n in (1, 3, 5)))


def set_cell(shape, name, formula):
    shape.CellsU(name).FormulaU = str(formula)


class Diagram:
    def __init__(self, doc, first, name, title, note, footer, tag):
        self.page = doc.Pages.Item(1) if first else doc.Pages.Add()
        self.page.Name = name
        self.font = next(f.ID for f in doc.Fonts if f.Name == "微软雅黑")
        self.nodes = {}
        for key, val in [("PageWidth", "12.2 in"), ("PageHeight", "6.9 in")]:
            set_cell(self.page.PageSheet, key, val)
        self.text(0.42, 6.42, 11.3, 0.25, tag, 10, "#33746B", True)
        self.text(0.42, 5.89, 11.3, 0.51, title, 25, INK, True)
        self.text(0.42, 5.47, 11.45, 0.40, note, 12, "#486070")
        self.page.DrawLine(0.43, 5.41, 11.77, 5.41).CellsU("LineColor").FormulaU = rgb(
            "#BFCFD6"
        )
        self.text(0.43, 0.19, 11.45, 0.52, footer, 10, "#486070")

    def text(self, x, y, w, h, value, size, color, bold=False):
        s = self.page.DrawRectangle(x, y, x + w, y + h)
        for key, val in [
            ("FillPattern", 0),
            ("LinePattern", 0),
            ("Para.HorzAlign", 0),
            ("LeftMargin", 0),
            ("RightMargin", 0),
            ("TopMargin", 0),
            ("BottomMargin", 0),
            ("Char.Font", self.font),
            ("Char.AsianFont", self.font),
            ("Char.Size", f"{size} pt"),
            ("Char.Color", rgb(color)),
            ("Char.Style", int(bold)),
        ]:
            set_cell(s, key, val)
        s.Text = value
        return s

    def node(self, row, col, family, heading, detail):
        x, y = 0.43 + col * 2.90, 4.08 - row * 1.52
        w, h = 2.63, 1.03
        s = self.text(x, y, w, h, heading + "\n" + detail, 10.4, INK)
        fill, border = COLORS[family]
        for key, val in [
            ("FillPattern", 1),
            ("FillForegnd", rgb(fill)),
            ("LinePattern", 1),
            ("LineColor", rgb(border)),
            ("LineWeight", ".85 pt"),
            ("Rounding", ".080 in"),
            ("LeftMargin", ".10 in"),
            ("RightMargin", ".07 in"),
        ]:
            set_cell(s, key, val)
        chars = s.Characters
        chars.Begin, chars.End = 0, len(heading)

        # The Characters API takes point size; style bit 0 selects bold.
        prop = chars._oleobj_.GetIDsOfNames("CharProps")
        chars._oleobj_.Invoke(prop, 0, pythoncom.DISPATCH_PROPERTYPUT, 0, 2, 1)
        chars._oleobj_.Invoke(prop, 0, pythoncom.DISPATCH_PROPERTYPUT, 0, 7, 13)
        self.nodes[row, col] = s

    def edge(self, a, b):
        first, last = self.nodes[a], self.nodes[b]
        horizontal = a[0] == b[0]
        beg, end = ((1, 0.5), (0, 0.5)) if horizontal else ((0.5, 0), (0.5, 1))
        line = self.page.DrawLine(0, 0, 1, 0)
        line.CellsU("BeginX").GlueToPos(first, *beg)
        line.CellsU("EndX").GlueToPos(last, *end)
        for key, val in [
            ("LineColor", rgb("#526D7D")),
            ("LineWeight", "1.0 pt"),
            ("EndArrow", 4),
            ("EndArrowSize", 1),
        ]:
            set_cell(line, key, val)

    def populate(self, rows, vertical=(), omit=()):
        for r, items in enumerate(rows):
            for c, (family, title, body) in enumerate(items):
                self.node(r, c, family, title, body)
        for r, items in enumerate(rows):
            for c in range(len(items) - 1):
                if (r, c) not in omit:
                    self.edge((r, c), (r, c + 1))
        for r, c in vertical:
            self.edge((r, c), (r + 1, c))


def build():
    OUT.mkdir(exist_ok=True, parents=True)
    app = com.DispatchEx("Visio.InvisibleApp")
    app.AlertResponse = 7
    doc = app.Documents.Add("")
    try:
        g = Diagram(
            doc,
            True,
            "方法总览",
            "从多视角观测到可编辑的物理部件",
            "三行分别给出批量估重、表面赋值和显式体域的输入与产物。第 2–4 页展开各条路径。",
            "ABO-500 的质量评测覆盖 500 件。显式内构、局部编辑和动力学各自使用具备相应几何的验证实例。",
            "PartPhy3D     /     方法总览",
        )
        g.populate(
            [
                [
                    ("input", "物体观测", "标定图像、相机参数\n按任务选帧"),
                    ("geometry", "几何与构造", "六视图重建外形\n三视图估尺度及构件"),
                    ("physics", "整件质量估计", "逐构件积分\n与图像质量估计融合"),
                    (
                        "result",
                        "ABO-500 质量结果",
                        "逐件质量、分支及回退记录\n训练 300 · 验证 100 · 测试 100",
                    ),
                ],
                [
                    ("input", "表面观察", "第 00、10、20 帧\n候选部件与可投影网格"),
                    (
                        "material",
                        "掩码投到网格",
                        "部件定位、SAM2 掩码\n深度检验、逐面投票",
                    ),
                    (
                        "material",
                        "表面材料与先验",
                        "合并同材票、受限补全\n保留部件 ID 与未知面",
                    ),
                    (
                        "result",
                        "跨视角表面图",
                        "材质、密度和模量图\n覆盖与材质 IoU 分别核对",
                    ),
                ],
                [
                    (
                        "input",
                        "可编辑实例",
                        "已绑定部件的米制几何\n及带来源的内部构造假设",
                    ),
                    (
                        "physics",
                        "互斥的材料体域",
                        "外壳、内芯、支撑与空腔\n按空间交叠进行扣除",
                    ),
                    (
                        "physics",
                        "体域积分或粒子化",
                        "几何矩供局部编辑\n适应界面的粒子供 MPM",
                    ),
                    (
                        "result",
                        "编辑与运动",
                        "质量、质心、惯量变化\n刚体轨迹或软体压缩",
                    ),
                ],
            ]
        )
        g = Diagram(
            doc,
            False,
            "批量质量",
            "一件物体的质量怎样算出来",
            "结构支路处理几何和材料；图像支路提供统计尺度上的质量参照。两路都保留原始估计。",
            "封存记录中，499 件有结构质量，1 件回退到图像质量。表面分割和 MPM 没有作为这 500 件估重的必经步骤。",
            "PartPhy3D     /     批量质量",
        )
        g.populate(
            [
                [
                    (
                        "input",
                        "六张标定观测",
                        "按六个目标方位无放回选帧\n保留真实相机内外参",
                    ),
                    (
                        "geometry",
                        "InstantMesh 重建",
                        "固定编码与重建权重\n相似变换还原相机坐标",
                    ),
                    (
                        "geometry",
                        "面积与有向体积",
                        "检查闭合性和面绕向\n输出相机单位下面积与体积",
                    ),
                ],
                [
                    (
                        "input",
                        "第 00、08、20 帧",
                        "同一物体的三张 RGB\n保留图像与推断记录",
                    ),
                    (
                        "material",
                        "Qwen 构造提议",
                        "最长边、材料与部件角色\n面积份额、体积份额、厚度",
                    ),
                    (
                        "physics",
                        "构造积分",
                        "按最长边恢复米制尺度\n区分实心、薄壳与部分填充",
                    ),
                    ("physics", "结构质量 P", "累加各部件 ρV\n覆盖层按厚度修正"),
                ],
                [
                    (
                        "input",
                        "第 00、08、20 帧",
                        "与构造提议共用图像\n训练集提供目录质量监督",
                    ),
                    ("geometry", "图像特征", "冻结 CLIP ViT-B/16\n归一化后跨视图平均"),
                    (
                        "physics",
                        "图像质量 Mᵢ",
                        "岭回归拟合对数质量\n在验证集选择正则强度",
                    ),
                    ("result", "最终质量", "对数比值经 tanh 限幅\n结构无效时输出 Mᵢ"),
                ],
            ],
            vertical=[(0, 2), (1, 3)],
        )
        g = Diagram(
            doc,
            False,
            "表面材质",
            "从二维区域证据到三角面物性",
            "一个网格面保存哪些视图支持它、属于哪个部件、有哪些候选材料，以及最终是否赋值。",
            "覆盖分解为几何命中率 × 命中后的赋值率。PhysSeg 的材料 IoU 单独检验预测类别是否与人工标注一致。",
            "PartPhy3D     /     表面材料",
        )
        g.populate(
            [
                [
                    ("input", "标定的三张 RGB", "第 00、10、20 帧\n已知投影与前景"),
                    (
                        "material",
                        "部件与候选材料",
                        "VLM 提议部件并定位\nSAM2 提取对应掩码",
                    ),
                    (
                        "material",
                        "多视角面投票",
                        "深度筛掉遮挡面\n各视图按面内像素归一",
                    ),
                    (
                        "result",
                        "逐面证据",
                        "保留部件 ID、票数和来源\n可检查冲突、遗漏与遮挡",
                    ),
                ],
                [
                    ("input", "逐面票与几何邻接", "同一材料的部件票合并\n空掩码弃权"),
                    (
                        "material",
                        "高置信种子",
                        "法向约束下局部平滑\n分差太小时保留未知",
                    ),
                    (
                        "material",
                        "受限补全",
                        "沿面邻接最短路径补全\n结合颜色与距离补未知处",
                    ),
                    (
                        "result",
                        "网格表面材料",
                        "按面输出材料与未知标签\n部件身份单独保留",
                    ),
                ],
                [
                    (
                        "input",
                        "材料类别及物性库",
                        "部件或材料的参数先验\n摩擦需指定接触配对",
                    ),
                    (
                        "physics",
                        "赋予物性",
                        "密度 ρ、杨氏模量 E 等\n记录参数值及其来源",
                    ),
                    (
                        "geometry",
                        "向指定视角投影",
                        "统一相机、光栅与前景\n未知区域显式显示",
                    ),
                    (
                        "result",
                        "属性图与误差核查",
                        "覆盖检查几何与赋值\n标注检查材质正确性",
                    ),
                ],
            ]
        )
        g = Diagram(
            doc,
            False,
            "编辑与动力学",
            "一次部件换材怎样传到整件物体",
            "针对有显式体域的实例，在同一物体坐标中保存相的几何、材料、体积与部件身份。",
            "参考躺椅检验局部编辑，条纹坐垫检验软体响应，扫描锤检验刚体运动。三项实验分别记录几何及载荷条件。",
            "PartPhy3D     /     编辑与动力学",
        )
        g.populate(
            [
                [
                    (
                        "input",
                        "米制部件与构造",
                        "部件 ID 与几何对应\n给定壳厚、内核或支撑",
                    ),
                    ("physics", "材料域互斥化", "扣除已占体积\n划分壳、核与空腔"),
                    (
                        "physics",
                        "缓存几何矩",
                        "每相体积、一阶矩、惯量\n以密度加权求整件质量矩",
                    ),
                    (
                        "result",
                        "定点换材",
                        "目标相加上密度差乘几何矩\n与整件重新积分比对",
                    ),
                ],
                [
                    (
                        "input",
                        "闭合体域与参数",
                        "相界面、壳核、支撑域\n各相密度、模量、泊松比",
                    ),
                    (
                        "physics",
                        "自适应内部采样",
                        "界面处细分、内部可粗采样\n空腔跳过，记录粒子体积",
                    ),
                    ("physics", "MLS-MPM", "粒子质量 m = ρV\n本构应力与粒子–网格传递"),
                    (
                        "result",
                        "软体响应",
                        "固定初态、地面与几何\n比较压缩曲线与守恒残差",
                    ),
                ],
                [
                    (
                        "input",
                        "质量矩与碰撞几何",
                        "由显式域得到整件质量\n质心位置与质心处惯量",
                    ),
                    (
                        "physics",
                        "设置发射条件",
                        "固定重力、接触与冲量\n记录初始姿态及施力位置",
                    ),
                    (
                        "physics",
                        "MuJoCo 刚体模拟",
                        "解算姿态与平移\n同步跟踪质心及几何中心",
                    ),
                    (
                        "result",
                        "运动与编辑对照",
                        "比较旋转、几何中心轨迹\n分辨质量与惯量的影响",
                    ),
                ],
            ]
        )
        doc.SaveAs(str(OUT / "PartPhy3D_pipeline_zh.vsdx"))
        doc.ExportAsFixedFormat(1, str(OUT / "PartPhy3D_pipeline_zh.pdf"), 1, 0)
    finally:
        doc.Saved = True
        doc.Close()
        app.Quit()


if __name__ == "__main__":
    build()
