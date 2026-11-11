# 电解液溶剂化结构、离子输运、等变图网络与电池材料大语言模型文献综述（2020–2026）

> **调研范围与验证说明**：本报告围绕 **液态电解液溶剂化物理与机器学习（Cluster A）**、**周期性电极/固态电解质等变图神经网络与普适势函数（Cluster B）**、**材料与电池领域大语言模型及晶体-文本多模态模型（Cluster C）**、以及 **多组分混合配方置换不变表征学习（Cluster D）** 四大子方向，系统梳理并核验了 **40 篇（2020–2026 年）真实发表于同行评审期刊或 arXiv 的代表性文献**。基于上述文献体系，本文提炼了溶剂化与晶体输运任务中的核心物理-几何对齐准则，剖析了现有“标量配方预测—昂贵全原子模拟—纯文本序列化大模型”三者之间的空白地带，并针对 YLM 电解液模块（`src/toolkit/domains.py`）提出 5 项可直接落地的 $O(3)$ 等变与物理先验升级方案。

---

## 1. 40 篇核心文献全景总表（2020–2026）

下表按四大子簇列出全部 40 篇经检索核验的代表性文献，涵盖研究体系类型（液态混合配方 / 局域溶剂化壳 / 周期性晶体）、对称性与方向性建模机制（$O(3)$ / $SO(3)$ 等变、周期性边界条件 PBC、纯标量不变特征、纯文本序列化）及核心局限性。

| 编号 | 论文题目、作者与发表年份/期刊（或 arXiv 编号） | 体系类型 (System Type) | 对称性与方向性 (Symmetry & Directionality) | 核心机制与关键局限性 (Key Mechanism & Limitation) |
| :--- | :--- | :--- | :--- | :--- |
| **Cluster A** | **液态电解液机器学习、溶剂化壳物理（SSIP / CIP / AGG）与电导率/氧化还原基准（11 篇）** | | | |
| **1** | *Quantum chemical calculations of lithium-ion battery electrolyte and interphase species* (**LIBE**)<br>Spotte-Smith et al. (Persson Group), **2021**, *Scientific Data*, 8, 203 | 局域溶剂化团簇 / SEI 物种（>17,000 分子与络合物） | 3D DFT 几何输入；下游统计以标量热力学量为主 | 提供了 $\omega\text{B97X-V}$ 级别电解液与界面物种势能与自由能基准，但均为气相/隐式溶剂孤立团簇，缺乏显式体相多分子动态配位竞争与方向性脱壳轨迹。 |
| **2** | *The solvation structure, transport properties and reduction behavior of carbonate-based electrolytes of lithium-ion batteries*<br>Hou, Fong, Wang & Persson, **2021**, *Chemical Science*, 12, 14740–14751 | 液态混合体系 + 局域溶剂化壳（碳酸酯体系） | PBC 经典/从头算 MD 采样；输出标量配位数与扩散系数 | 系统关联了 SSIP/CIP/AGG 比例、自扩散系数与还原电位偏移，但完全依赖长时 MD + DFT 后处理，无法对新配方与任意空间方向做即时推理。 |
| **3** | *Tailoring electrolyte solvation for Li metal batteries cycled at ultra-low temperature*<br>Holoubek et al. (Ping Liu Group), **2021**, *Nature Energy*, 6, 303–313 | 局域溶剂化壳 + 电极界面（超低温锂金属电池） | 3D 离子-偶极配位物理分析；输出标量去溶剂化能垒 | 揭示溶剂强偶极向心锁定导致 SSIP 高去溶剂化能垒，弱溶剂化促进 CIP/AGG 降低脱壳阻力；但缺乏自动化等变机器学习模型直接预测 3D 去溶剂化逃逸矢量。 |
| **4** | *RedDB, a computational database of electroactive molecules for aqueous redox flow batteries*<br>Sorkun et al. (Er Group), **2022**, *Scientific Data*, 9, 718 | 液流电池电活性单分子（31,618 醌类/氮杂芳烃） | 2D 分子图 / 3D 构象标量量子化学描述符（Scalar-only） | 构建了氧化还原电位与机器学习水溶性高通量数据库，但仅刻画无限稀释单分子标量性质，缺失多组分盐-溶剂协同溶剂化壳与界面取向信息。 |
| **5** | *Data-driven electrolyte design for lithium metal anodes*<br>Kim, Oyakhire, Yu, Bao, Bent & Cui, **2023**, *PNAS*, 120(10), e2214357120 | 液态混合电解液配方（锂金属负极库仑效率） | 标量元素组成与手工分子描述符（Scalar-only） | 发现“降低溶剂氧含量”可削弱 $\text{Li}^+$ 溶剂化过强束缚并提升 CE 至 99.7%，但输入仅为粗粒度元素比例标量，完全丢失 3D 配位几何与偶极取向结构。 |
| **6** | *Electrostatic Potential as Solvent Descriptor to Enable Rational Electrolyte Design for Lithium Batteries*<br>Wu et al. (Xiangming He Group), **2023**, *Advanced Energy Materials*, 13(22), 2300259 | 电解液溶剂单分子 / 局域离子-溶剂配位 | 3D 分子表面静电势极值投影为标量 $(\text{ESP}_{\min}, \text{ESP}_{\max})$ | 用 $(\text{ESP}_{\min}, \text{ESP}_{\max})$ 二维图谱清晰区分强溶剂、弱溶剂与非溶剂稀释剂，但将 3D 各向异性静电场压缩为两个标量极值，忽略了多配体空间排斥与偶极矢量夹角。 |
| **7** | *SolvationAnalysis: A Python toolkit for understanding liquid solvation structure in classical molecular dynamics simulations*<br>Cohen, Fong, Hou & Persson, **2023**, *J. Open Source Softw.*, 8(84), 5183 | 液态电解液第一溶剂化壳（MD 轨迹解析） | 支持 PBC 最小镜像距离截断；输出标量配位数与物种分率 | 标准化了径向分布函数壳层截断、配位数（CN）及 SSIP/CIP/AGG 精确图论定义，但属于纯后处理统计脚本，依赖预先完成的昂贵 MD 轨迹，且不输出方向性张量序参量。 |
| **8** | *CALiSol-23: Experimental electrolyte conductivity data for various Li-salts and solvent combinations*<br>de Blasio, Elsborg, Vegge, Flores & Bhowmik, **2024**, *Scientific Data*, 11, 750 | 液态多组分混合配方（13,825 条实验电导率数据） | 纯配方标量表征（SMILES + 摩尔比 + 温度 $T$，Scalar-only） | 涵盖 14 种锂盐与 38 种溶剂的权威实验离子电导率基准，但仅有宏观标量标签 $\sigma(T)$，不含微观溶剂化笼 3D 几何、配位数或离子迁移方向真值。 |
| **9** | *A predictive machine learning force field framework for liquid electrolyte development* (**BAMBOO**)<br>Gong et al. (ByteDance Research), **2024**, arXiv:2404.07181 | 液态多组分电解液（支持最多 15 种化学组分体相模拟） | $O(3)$ 图等变 Transformer + PBC 体相长程静电修正 | 通过集成知识蒸馏与实验密度对齐实现高精度电导率/粘度 MD 预测，但获取宏观与溶剂化性质仍需运行纳秒级全原子 MD 积分，无法实现文本条件化单步方向推理。 |
| **10** | *The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models*<br>Levine et al. (Meta FAIR & LBNL), **2025**, arXiv:2505.08762 | 溶剂化金属离子团簇、液态电解液片段与反应复合物（~1 亿 DFT 构象） | $SO(3)$ / $O(3)$ 等变原子势训练基准（非周期团簇，无 PBC） | 首次在大规模 $\omega\text{B97M-V}$ 数据集中大规模覆盖 $\text{Li}^+/\text{Na}^+$ 显式溶剂化壳与电荷/自旋态，但聚焦于力/能量回归，缺少面向自然语言交互的溶剂化机理与去溶剂化方向读出层。 |
| **11** | *Open electrolyte database generated via an automated molecular dynamics simulation framework* (**OEDB**)<br>Nakamura, Takenaka, Tsuda & Yamada et al., **2026**, *npj Computational Materials*, 12 | 液态电解液多模态模拟库（MD 结构 + 扩散系数 + 拉曼光谱） | PBC 自动化 MD 生成；发布特征以标量输运量与 1D 谱图为主 | 构建了高通量自动化电解液 MD 多模态数据库，但下游预测仍停留于标量扩散系数与一维光谱拟合，未建立局域 3D 溶剂化笼各向异性张量表征。 |
| **Cluster B** | **周期性电极与固态电解质离子迁移的等变图网络与普适势函数（10 篇）** | | | |
| **12** | *A charge-density-based general cation insertion algorithm for generating new Li-ion cathode materials*<br>Shen, Horton & Persson, **2020**, *npj Computational Materials*, 6, 161 | 周期性嵌入式正极晶体（Materials Project Insertion Electrodes） | 3D PBC 电荷密度极小值位点搜索；输出嵌入位点与电压标量 | 利用 DFT 电荷密度拓扑自动发现周期性框架中的空位嵌入位点并构建 MP 嵌入电极库，但依赖静态 DFT 电子密度，未直接输出等变迁移过渡态方向矢量。 |
| **13** | *High-throughput screening platform for solid electrolytes combining hierarchical ion-transport prediction algorithms*<br>He et al. (Siqi Shi Group), **2020**, *Scientific Data*, 7, 151 | 周期性无机固态电解质晶体（>29,000 化合物筛查） | 3D PBC 几何空隙（CAVD）与键价位点能（BVSE）标量场 | 结合几何瓶颈半径与经验 BVSE 势场快速构建三维渗流网络与近似迁移势垒，但刚性晶格经验势忽略了晶格弛豫与局域电子极化，对快离子导体误报率较高。 |
| **14** | *A universal graph deep learning interatomic potential for the periodic table* (**M3GNet**)<br>Chen & Ong, **2022**, *Nature Computational Science*, 2, 718–728 | 周期性无机晶体（覆盖全周期表结构弛豫与声子/MD） | $E(3)$ 不变距离 + 三体键角特征 + PBC（无高阶不可约张量） | 引入连续可微三体交互处理周期表任意晶体结构弛豫，但仅使用标量距离与键角（$l=0$ 不变量），消息传递缺乏高阶等变球谐张量，对复杂过渡态力场精度受限。 |
| **15** | *CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling* (**CHGNet**)<br>Deng et al. (Ceder Group), **2023**, *Nature Machine Intelligence*, 5, 1031–1041 | 周期性电池电极与固态电解质晶体（MPtrj 预训练） | $E(3)$ 不变原子图 + 键角图 + PBC + 磁矩约束原子电荷态 | 通过磁矩监督显式推断过渡金属价态（如 $\text{Fe}^{2+}/\text{Fe}^{3+}$、$\text{Mn}^{3+}/\text{Mn}^{4+}$），极适合变价正极，但内部特征仍为标量不变表示，且获取迁移势垒仍需迭代运行 NEB/MD。 |
| **16** | *A foundation model for atomistic materials chemistry* (**MACE-MP-0**)<br>Batatia et al. (Csányi Group), **2024**, arXiv:2401.00096 | 周期性无机晶体、表面与分子液体（跨领域原子基础模型） | 高阶 $O(3)$ 等变消息传递（ACE 多体张量积，$l \ge 0$）+ PBC | 利用高体序等变球谐张量积在零样本/少样本下准确复现晶体力学、固态电解质扩散与液态水结构，但仅输出原子标量能与梯度力，不支持语言条件查询或显式通道张量读出。 |
| **17** | *EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations*<br>Liao, Wood, Das & Smidt, **2024**, *ICLR 2024* (arXiv:2306.12059) | 周期性催化表面、无机晶体与三维分子体系 | 高阶 $SO(3)$ / $O(3)$ 等变 Transformer（$e\text{SCN}$ 卷积，$L_{\max}$ 可扩展至 6）+ PBC | 用 $SO(2)$ 线性旋转卷积将高阶球谐张量积复杂度从 $O(L_{\max}^6)$ 降至 $O(L_{\max}^3)$，方向表达能力极强，但参数与算力开销大，且完全没有自然语言对齐接口。 |
| **18** | *Scalable Parallel Algorithm for Graph Neural Network Interatomic Potentials in Molecular Dynamics Simulations* (**SevenNet**)<br>Park, Kim, Hwang & Han, **2024**, *J. Chem. Theory Comput.*, 20(11), 4857–4868 | 大尺度周期性固态电解质与界面体系（多 GPU 并行 MD） | 严格 $E(3)$ 等变图卷积（基于 NequIP 球谐不可约表示）+ PBC 空间域分解 | 解决了多层消息传递等变 GNN 在大规模 LAMMPS 并行模拟中的通信瓶颈，但仍以长时 MD 积分方式间接统计扩散系数，无法单步推理局部迁移矢量。 |
| **19** | *MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures*<br>Yang et al. (Microsoft Research), **2024**, arXiv:2405.04967 | 周期性晶体、熔体与宽温压材料体系（0–5000 K, 0–1000 GPa） | $E(3)$ 不变/等变图编码器（M3GNet / Graphormer3D 变体）+ PBC | 通过主动学习覆盖极端温压相空间，显著提升热力学与扩散稳定性，但仍是黑盒势能面回归器，缺乏与化学文本推理和配方层表征的耦合。 |
| **20** | *Orb: A Fast, Scalable Neural Network Potential*<br>Neumann et al. (Orbital Materials), **2024**, arXiv:2410.22570 | 大规模周期性无机晶体（Matbench Discovery 高通量筛选） | 非严格数据增强近似旋转不变 GNN（光滑径向/矢量注意力）+ PBC | 放弃严格 Clebsch–Gordan 张量积以换取比传统等变网络快数倍的吞吐量，但由于破坏了精确 $O(3)$ 群等变性，旋转下能量/力存在非零数值漂移，不适合精确协变矢量读出。 |
| **21** | *Benchmarking machine learning models for predicting lithium ion migration* (**LiTraj**)<br>Dembitskiy et al., **2025**, *npj Computational Materials*, 11, 131 | 周期性含锂无机晶体（13,000 渗流阈值 + 122,000 迁移势垒 + 1,700 DFT-NEB 轨迹） | PBC 晶体图输入；评测标量势垒回归 GNN 与 uMLIP-NEB | 构建了迄今最大规模 BVSE + DFT-NEB 锂离子迁移基准，证明 BVSE 预训练转 DFT 微调可大幅提速，但直接预测模型仅输出标量势垒 $E_a$，未预测三维跃迁鞍点切向矢量。 |
| **Cluster C** | **材料与电池领域大语言模型及晶体-文本多模态模型（10 篇）** | | | |
| **22** | *MatSciBERT: A Materials Domain Language Model for Text Mining and Information Extraction*<br>Gupta, Zaki, Krishnan & Mausam, **2022**, *npj Computational Materials*, 8, 197 | 无机材料科技文献语料（15 万篇材料论文预训练） | 纯文本编码器（Text-only，无任何 3D 几何或对称性） | 在材料实体识别与关系抽取上显著优于通用 SciBERT，但完全不接收三维原子坐标、晶胞参数或分子拓扑，无法感知空间配位与对称性。 |
| **23** | *BatteryBERT: A Pretrained Language Model for Battery Database Enhancement*<br>Huang & Cole, **2022**, *J. Chem. Inf. Model.*, 62(24), 6365–6377 | 电池领域全文文献库（>40 万篇电池论文预训练） | 纯文本编码器（Text-only，无结构等变性） | 面向电池正负极、电解液文献问答与自动数据库扩充构建专用词表与编码器，但仅停留在文献文本挖掘，缺乏对分子/晶格三维物理规律的定量推理能力。 |
| **24** | *CrysMMNet: Multimodal Representation for Crystal Property Prediction*<br>Das et al. (Ganguly Group), **2023**, *Proceedings of UAI 2023* (PMLR 216:507–517) | 周期性无机晶体 + 晶体结构文本描述（多模态融合） | PBC 标量晶体图（距离/键角不变 GNN）+ 文本编码器融合；输出标量属性 | 首次将局部键角晶体图特征与全局文本语义拼接以提升形成能与带隙预测，但图分支仅为标量不变表示，跨模态融合后彻底坍缩为 $l=0$ 标量，无方向性输出能力。 |
| **25** | *DARWIN Series: Domain Specific Large Language Models for Natural Science*<br>Xie et al. (Hoex Group), **2023**, arXiv:2308.13565 | 材料、化学与物理科学问答及性质指令微调数据集 | 纯自回归文本生成（Text-only，SMILES/化学式字符串） | 通过自动化科学指令构造微调开源大模型完成材料性质分类与逆向设计问答，但所有分子与晶体均退化为一维字符串，丧失三维空间度量与等变约束。 |
| **26** | *LLM-Prop: Predicting Physical And Electronic Properties Of Crystalline Solids From Their Text Descriptions*<br>Rubungo, Arnold, Rand & Dieng, **2023/2025**, arXiv:2310.14029 (*npj Comput. Mater.* 2025) | 周期性晶体（Robocrystallographer 自动生成的结构描述文本） | 纯文本编码（将键长、键角、配位多面体文字化，Text-only） | 证明仅靠结构自动转换的自然语言描述即可高精度预测带隙与晶胞体积，但文字化过程丢失了精确三维张量方位关系，无法保证旋转协变或输出 3D 物理矢量。 |
| **27** | *Fine-Tuned Language Models Generate Stable Inorganic Materials as Text* (**Crystal-Text-LLM**)<br>Gruver, Sriram, Madotto, Wilson, Zitnick & Ulissi, **2024**, *ICLR 2024* (arXiv:2402.04379) | 周期性无机晶体生成（LLaMA-2 微调生成晶格长度/角度与分数坐标文本） | 1D 文本序列化（仅靠训练期随机平移/旋转数据增强近似对称性） | 直接用自回归 LLM 吐出十进制晶格参数与原子坐标并生成热力学亚稳新材料，但显式破坏了精确 $SE(3)$ 与置换不变性，且对连续几何微扰高度敏感。 |
| **28** | *Crystal structure generation with autoregressive large language modeling* (**CrystaLLM**)<br>Antunes, Butler & Grau-Crespo, **2024**, *Nature Communications*, 15, 10570 | 周期性无机晶体生成（数百万 CIF 文件从头预训练 GPT 架构） | 1D CIF 字符串自回归生成（依赖空间群符号隐式约束，无 $O(3)$ 等变层） | 在百万级无机晶体 CIF 语料上从零训练语言模型以学习对称空间群与配位语法，但模型内部无三维欧氏距离或球谐张量计算，无法对给定局部构象做方向性物理探测。 |
| **29** | *HoneyComb: A Flexible LLM-Based Agent System for Materials Science*<br>Zhang, Song, Hou, Miret & Liu, **2024**, *EMNLP 2024 Findings* (arXiv:2409.00135) | 材料科学知识库（MatSciKB）与外部计算工具调用智能体 | 纯文本控制流智能体（Text-only orchestration） | 通过结构化知识检索（MatSciKB）与动态工具中心（ToolHub）显著抑制材料问答幻觉，但 LLM 本身没有原生三维几何感知层，只能被动拼接外部脚本的标量返回值。 |
| **30** | *MatExpert: Decomposing Materials Discovery by Mimicking Human Experts*<br>Ding, Miret & Liu, **2024**, arXiv:2410.21317 | 无机功能材料检索、过渡修改与从头结构生成流水线 | 多阶段文本推理 + CIF 字符串生成（无内生 $O(3)$ 等变特征桥） | 模仿人类专家将材料发现拆解为检索、机制分析、结构生成与 DFT/MLIP 校验四步，但推理链与三维几何结构之间仅通过离散 CIF 文本交互，无法传递连续高阶不可约张量。 |
| **31** | *Expert-Guided LLM Reasoning for Battery Discovery: From AI-Driven Hypothesis to Synthesis and Characterization* (**ChatBattery**)<br>Liu, Xu, Ai, Li, Bengio & Guo, **2025**, arXiv:2507.16110 | 锂离子电池正极材料发现与湿法合成闭环验证 | 领域规则引导的纯文本化学式推理（Text-only composition reasoning） | 结合专家检索代理在正极材料组分空间中完成假设生成并经实验合成验证高容量新组分，但仅在化学计量式层面推理，完全未建模电解液三维溶剂化笼或晶内各向异性通道。 |
| **Cluster D** | **多组分混合配方表征学习与物理约束电导率预测（9 篇）** | | | |
| **32** | *Autonomous optimization of non-aqueous Li-ion battery electrolytes via robotic experimentation and machine learning coupling*<br>Dave, Mitchell, Burke, Lin, Whitacre & Viswanathan, **2022**, *Nature Communications*, 13, 5454 | 非水液态多组分混合电解液（Clio 机器人实验台闭环寻优） | 组分质量/摩尔分数向量输入（高斯过程蜻蜓贝叶斯优化，Scalar-only） | 实验机器人与贝叶斯优化闭环在数日内寻得快充高电导率碳酸酯/酯混合配方，但仅在固定组分池单纯形上插值，无分子图或 3D 溶剂化结构表征，无法跨未知分子外推。 |
| **33** | *Chemistry-Informed Machine Learning for Polymer Electrolyte Discovery*<br>Bradford et al. (Gómez-Bombarelli & Shao-Horn), **2023**, *ACS Central Science*, 9(2), 206–216 | 多组分聚合物-锂盐电解质混合体系 | 2D MPNN 分子图 + 组分分子量/盐浓比标量融合 + Arrhenius 物理读出头 | 将 Arrhenius 温度激活定律嵌入 MPNN 读出层预测数千种聚合物-阴离子组合电导率，但输入仅为 2D 拓扑图，无法刻画三维配位笼空间畸变与方向性链段协同输运。 |
| **34** | *SolvBERT for solvation free energy and solubility prediction: a demonstration of an NLP model for predicting the properties of molecular complexes*<br>Yu et al. (An Su Group), **2023**, *Digital Discovery*, 2, 409–421 | 溶质–溶剂二元分子复合物（溶剂化自由能与溶解度） | 拼接 SMILES 字符串（Text-only，不满足严格多组分集合置换不变性） | 在百万级量子化学溶质-溶剂对上无监督预训练 RoBERTa 表征，但采用分隔符拼接一维 SMILES，对组分排列顺序敏感，且缺乏连续摩尔分数加权与三维离子-偶极取向感知。 |
| **35** | *Differentiable Modeling and Optimization of Non-Aqueous Li-Based Battery Electrolyte Solutions Using Geometric Deep Learning* (**DiffMix**)<br>Zhu, Ramsundar, Viswanathan et al., **2024**, *Nature Communications*, 15 (arXiv:2310.03047) | 非水锂电池多组分液态电解液（电导率与热力学过量性质端到端梯度优化） | 2D 分子图编码 + 严格组分置换不变聚合 + VTF / Redlich–Kister 可微物理方程层 | 将 Vogel–Tammann–Fulcher（VTF）输运方程与热力学混合活度定律做成端到端可微网络用于配方梯度反演，但分子与混合物表征均为 $l=0$ 标量，无显式 3D 溶剂化壳等变几何。 |
| **36** | *Thermodynamics-consistent graph neural networks*<br>Rittig & Mitsos, **2024**, *Chemical Science*, 15(44), 18504–18512 | 二元/多元液态分子混合物（过量吉布斯自由能与活度系数） | 2D 分子图 + 置换不变混合池化 + 自动微分强制满足 Gibbs–Duhem 热力学一致性 | 通过预测标量过量吉布斯自由能 $g^E$ 并对摩尔分数求导严格保证活度系数热力学自洽与纯组分极限，但仅针对中性有机液体宏观活度，未涉及带电离子配位壳与三维矢量场。 |
| **37** | *Learning Molecular Mixture Property Using Chemistry-Aware Graph Neural Network* (**MolSets**)<br>Zhang, Lai, Chen, Manthiram, Rondinelli & Chen, **2024**, *PRX Energy*, 3, 023006 | 多组分液态电池电解液与离子液体混合物 | 2D 分子 GNN + Deep Sets 严格集合置换不变注意力池化 + 摩尔分数权重门控 | 解决了多组分任意数量溶剂/锂盐配方的置换不变性与零浓度极限连续性问题，但完全基于 2D 化学连接图，无法区分构象异构、空间位阻及 $\text{Li}^+$ 第一配位壳偶极对齐效应。 |
| **38** | *Uni-ELF: A Multi-Level Representation Learning Framework for Electrolyte Formulation Design*<br>Zeng et al. (Weinan E & Linfeng Zhang Group), **2024**, arXiv:2407.06152 | 多组分液态电解液配方（锂离子与锂硫电池电导率、粘度、CE） | 单分子 3D $SE(3)$ 不变距离矩阵（Uni-Mol）+ 混合物 MD 径向分布函数 $g(r)$ 监督预训练 | 利用两阶段预训练将单分子 3D 构象特征对齐至混合体系一维径向分布函数（RDF），显著提升配方性质预测，但 RDF 将三维各向异性溶剂化壳球面积分为一维标量曲线，彻底抹平了角度与矢量不对称性。 |
| **39** | *Chemical foundation model-guided design of high ionic conductivity electrolyte formulations*<br>Zohair, Sharma, Soares et al. (IBM Research), **2025**, *npj Computational Materials*, 11 | 非水多组分锂电液态配方（基础化学语言模型引导实验筛选） | 1D SMILES 化学语言基础模型（MoLFormer / SMI-TED）嵌入 + 组成加权聚合（Scalar-only） | 利用亿级分子预训练的化学 Transformer 提取组分嵌入并指导高电导率新配方实验合成，但底层表征源于无三维坐标的字符串编码，无法提供微观脱壳路径解释。 |
| **40** | *A Unified Predictive and Generative Solution for Liquid Electrolyte Formulation*<br>Yang et al. (Wen Yan Group, ByteDance Seed), **2025/2026**, *Nature Machine Intelligence* (arXiv:2504.10008) | 多组分液态电解液全流程正向性质预测与条件逆向配方生成 | 置换不变配方集合 Transformer（分子编码 + 浓度连续嵌入，Scalar-only） | 首次在同一架构下统一多组分电解液正向性质回归与条件配方生成（给定目标电导率/CE 直接生成分子与配比），但仍工作于配方级标量潜空间，缺失局域 3D 溶剂化笼与晶格界面的等变几何桥梁。 |

---

## 2. 四大子方向技术脉络深度述评

### 2.1 Cluster A：液态电解液溶剂化壳物理（SSIP / CIP / AGG）与力场/电导率基准（文献 #1–#11）
2020–2026 年间，液态电解液设计的物理核心已从传统的宏观介电常数（$\varepsilon$）与粘度（$\eta$）经验权衡，全面转向**第一溶剂化鞘（Primary Solvation Sheath）的微观配位工程**：
1. **配位态谱系与去溶剂化动力学瓶颈**：Persson 团队（Hou et al., 2021 [#2]; Cohen et al., `SolvationAnalysis`, 2023 [#7]）与 Ping Liu 团队（Holoubek et al., *Nature Energy* 2021 [#3]）确立了定量的微观物种分类范式——溶剂分离离子对（**SSIP**）、接触离子对（**CIP**）与离子聚集体（**AGG**）。在快充与超低温（$-40^\circ\text{C} \sim -60^\circ\text{C}$）工况下，高极性环状碳酸酯（如 EC）因其大偶极矩与 $\text{Li}^+$ 形成强向心偶极锁定的刚性 SSIP 笼，导致极高的界面去溶剂化能垒（$\Delta G_{\text{desolv}}$）；而通过局部高浓（LHCE）、弱溶剂化醚类或低氧含量溶剂设计（Kim & Cui et al., *PNAS* 2023 [#5]）驱使阴离子（如 $\text{FSI}^-$）进入第一配位壳形成 CIP/AGG，不仅打破了纯溶剂偶极对称笼的刚性束缚，还优先还原生成富含 $\text{LiF}$ 的无机 SEI。
2. **量子化学描述符与大规模数据集演进**：从单分子/孤立团簇量子化学库（`LIBE` [#1] 含 >17,000 个电解液与 SEI 裂解物种；`RedDB` [#4] 含 31,618 个氧化还原活性有机分子；Wu et al. [#6] 提出的分子表面静电势极值对 $(\text{ESP}_{\min}, \text{ESP}_{\max})$），发展到宏观实验电导率基准库 `CALiSol-23` [#8]（13,825 个实验点）与高通量自动化 MD 数据库 `OEDB` [#11]（2026），再到覆盖近亿级显式金属离子溶剂化团簇 DFT 计算的 `OMol25` [#10]（Meta FAIR & LBNL, 2025）与专用等变机器学习力场 `BAMBOO` [#9]（ByteDance, 2024）。
3. **核心痛点**：尽管 `BAMBOO` 与 `OMol25` 训练的等变势函数已能以接近 DFT 的精度驱动体相电解液 MD，但提取 SSIP/CIP/AGG 比例、去溶剂化能垒及配位不对称性仍必须执行成千上万步分子动力学时间积分，计算延迟高达分钟至小时级，无法作为快响应可微模块直接服务于自然语言交互式机理探针。

### 2.2 Cluster B：周期性电极与固态电解质等变图网络及离子迁移势垒预测（文献 #12–#21）
在正极材料与固态电解质（SSE）晶格中，$\text{Li}^+$ 的嵌入与扩散受控于三维周期性晶体势能面（PES）中的鞍点过渡态与各向异性渗流通道：
1. **从经验几何/键价景观到普适原子基础模型（uMLIPs）**：早期高通量筛选依赖 DFT 电荷密度极小值插锂算法（Shen, Horton & Persson, 2020 [#12]）或经验键价位点能/空隙几何分析（He & Shi et al., `BVSE/CAVD`, 2020 [#13]）。随后，普适晶体图势函数经历了从**标量三体不变架构**（`M3GNet` [#14]、电荷/磁矩感知的 `CHGNet` [#15] 与跨温压的 `MatterSim` [#19]）向**高阶 $O(3)/SO(3)$ 球谐不可约张量等变架构**（`MACE-MP-0` [#16] 的高体序 ACE 张量积、`EquiformerV2` [#17] 的 $SO(2)$ 高阶球谐注意力、以及支持多 GPU 空间并行 MD 的 `SevenNet` [#18]）的跃迁。此外，`Orb` [#20] 探索了数据增强非严格等变路线以提升吞吐速度，但牺牲了精确旋转群等变性。
2. **迁移路径基准与剩余鸿沟**：2025 年发布的 `LiTraj` 基准（Dembitskiy et al., *npj Comput. Mater.* 2025 [#21]）整合了 12.2 万个迁移势垒与 1,700 条 DFT-NEB 轨迹，证实 uMLIP 微调结合 BVSE 预训练可大幅压缩筛查成本。然而，现有所有晶体 GNN 要么**仅预测标量激活能 $E_a$**（丢失了晶格内的三维跃迁方向与各向异性主轴），要么**必须显式构造初末态超胞并迭代运行数十步 NEB 弹性带松弛**才能获得路径切向矢量。

### 2.3 Cluster C：材料与电池大语言模型及晶体-文本多模态架构（文献 #22–#31）
近年材料与电池领域的大语言模型沿着三条技术路线演进：
1. **文献挖掘编码器与智能体工作流**：`MatSciBERT` [#22] 与 `BatteryBERT` [#23] 通过领域语料掩码预训练实现了高精度电池实体抽取；`DARWIN` [#25]、`HoneyComb` [#29]、`MatExpert` [#30] 与 `ChatBattery` [#31]（Liu, Bengio & Guo et al., 2025）进一步引入科学指令微调、知识库检索（MatSciKB）与专家分步推理，在正极化学式假设生成上取得了湿法实验验证突破。
2. **三维晶体结构的 1D 文本序列化生成**：`Crystal-Text-LLM`（Gruver et al., *ICLR* 2024 [#27]）与 `CrystaLLM`（Antunes et al., *Nat. Commun.* 2024 [#28]）直接将晶胞长度、夹角、空间群与分数坐标格式化为纯文本/CIF 字符串进行自回归建模；`LLM-Prop` [#26] 则借助 Robocrystallographer 将局部键长键角转写为英文段落以回归标量带隙。
3. **多模态浅层标量融合**：`CrysMMNet` [#24] 尝试拼接不变晶体 GNN 与文本特征，但输出端仍仅限于标量属性。
4. **核心痛点**：**没有任何一个现有电池/材料 LLM 保留了三维物理空间的 $O(3)$ 等变几何不可约表示（irreps）！** 将三维坐标拆解为十进制数字 token 或英文描述词，不仅彻底破坏了连续旋转协变性与周期性最小镜像拓扑，也使模型根本无法回答诸如*“在该局部溶剂化笼或正极晶胞中，$\text{Li}^+$ 沿哪个三维空间矢量脱壳/跃迁阻力最小？”*这类方向性物理问题。

### 2.4 Cluster D：多组分混合配方置换不变表征与热力学/输运先验融合（文献 #32–#40）
与单分子药物发现不同，真实电池电解液是由锂盐、主溶剂、共溶剂/稀释剂和成膜添加剂组成的**连续配比无序集合 $\mathcal{F} = \{(m_i, c_i, \tau_i)\}_{i=1}^K$**：
1. **置换不变性与零浓度极限连续性**：早期拼接 SMILES 的语言模型（如 `SolvBERT` [#34]）不满足多组分输入顺序置换不变性；机器人闭环优化（Dave & Viswanathan et al., 2022 [#32]）则局限于固定组分列表。`MolSets`（Zhang et al., *PRX Energy* 2024 [#37]）通过分子图 GNN + 摩尔分数门控 Deep Sets 注意力池化，严格满足了集合置换不变性 $\Phi(\pi\cdot\mathcal{F})=\Phi(\mathcal{F})$ 与零浓度连续极限 $\lim_{c_k\to 0}\Phi(\mathcal{F}\cup\{(m_k,c_k)\})=\Phi(\mathcal{F})$；IBM 团队（Zohair & Soares et al., 2025 [#39]）与 ByteDance Seed 团队（Yang et al., *Nat. Mach. Intell.* 2025/2026 [#40]）进一步将其扩展至化学语言基础模型引导的高电导率筛选与统一正向预测/逆向配方生成。
2. **热力学与输运物理方程内嵌**：Bradford & Shao-Horn et al. [#33] 在 MPNN 读出头嵌入 Arrhenius 方程；`DiffMix`（Zhu, Ramsundar & Viswanathan, *Nat. Commun.* 2024 [#35]）内嵌非晶液体 VTF 方程与 Redlich–Kister 多项式以支持对配比的梯度反演；Rittig & Mitsos [#36] 利用自动微分强制满足 Gibbs–Duhem 活度一致性；`Uni-ELF`（Zeng, E & Zhang, 2024 [#38]）则引入 Uni-Mol 3D 单分子预训练并以 MD 径向分布函数 $g(r)$ 作为混合物中间监督信号。
3. **核心痛点**：即便是目前最先进的 `Uni-ELF` [#38] 与统一配方生成模型 [#40]，在混合物层级也仅保留了球面平均后的径向标量曲线 $g(r)$ 或 $l=0$ 标量潜向量，完全缺失三维第一溶剂化笼中配体偶极取向 $\cos\theta$、笼偶极不对称矢量及空间逃逸通道的显式几何表征。

---

## 3. 物理与几何对齐规范综合分析（Alignment Synthesis）

为确保电解液溶剂化壳与周期性电极/固态电解质建模在物理化学与群论几何上严格自洽，基于上述 40 篇文献的标准定义，任何跨尺度多模态模型必须同时对齐以下 **五项核心准则**：

### 3.1 配位数（Coordination Number, CN）与可微径向壳层截断
在经典与机器学习 MD 分析（如 Persson 组 `SolvationAnalysis` [#7] 与 `Uni-ELF` [#38]）中，$\text{Li}^+$ 第一溶剂化壳通常由径向分布函数 $g_{\text{Li-X}}(r)$ 的第一极小值半径 $r_c$ 界定（对于常见碳酸酯/醚类的羰基氧或醚氧 $\text{Li}^+\text{--O}$，典型值 $r_c \approx 2.6\text{--}2.8\,\text{\AA}$）。硬阶跃函数 $\Theta(r_c - r_{ij})$ 会在边界附近引发微小热涨落下的特征跳变与梯度消失，因此在神经网络几何编码中应采用平滑余弦包络权重：
$$
w(r_{ij}) = \frac{1}{2}\left[\cos\!\left(\frac{\pi\, r_{ij}}{r_c}\right) + 1\right]\mathbb{I}(r_{ij} < r_c)
$$
据此可分别定义平滑可微的**溶剂配位数** $\text{SCN} = \sum_{s \in \mathcal{S}_{\text{solv}}} \sum_{a \in s} w(r_{\text{Li}, a})$ 与**阴离子配位数** $\text{ACN} = \sum_{m \in \mathcal{S}_{\text{anion}}} \sum_{b \in m} w(r_{\text{Li}, b})$，其和即为总有效配位数 $\text{CN}_{\text{total}} = \text{SCN} + \text{ACN}$（典型四面体至五配位区间为 $4.0 \sim 5.0$）。

### 3.2 SSIP / CIP / AGG 拓扑物种分类准则
对齐 Borodin、Persson（[#2, #7]）与 Liu/Cui（[#3, #5]）文献体系的图论定义，构建以阳离子 $\text{Li}^+$ 与阴离子 $\text{A}^-$ 为两类核心节点的二分连接图（当阳离子 $i$ 与阴离子 $m$ 满足最近配位原子距离 $< r_c$ 时连边 $A_{im}=1$）：
- **SSIP（溶剂分离离子对）**：阳离子 $i$ 的第一配位壳内不含任何阴离子，即 $N_{\text{anion}}(i) = \sum_m A_{im} = 0$；
- **CIP（接触离子对）**：阳离子 $i$ 至少配位一个阴离子（$N_{\text{anion}}(i) \ge 1$），且其所配位的每一个阴离子 $m$ 均仅与该单一 $\text{Li}^+$ 配位（即阴离子的阳离子桥接度 $N_{\text{Li}}(m) = \sum_{i'} A_{i'm} = 1$）；
- **AGG（离子聚集体）**：阳离子 $i$ 至少配位一个阴离子（$N_{\text{anion}}(i) \ge 1$），且存在至少一个配位阴离子同时桥接两个及以上阳离子（$\max_{m: A_{im}=1} N_{\text{Li}}(m) \ge 2$），形成 $\text{Li}^+\text{--Anion--Li}^+$ 团簇或渗流网络。

### 3.3 配体偶极–离子径向对齐度（Ligand Dipole–Ion Radial Alignment, $\cos\theta$）
如 Holoubek et al. [#3] 与 Wu et al. [#6] 所示，$\text{Li}^+$ 去溶剂化能垒不仅取决于配位数，更直接受控于**离子–偶极相互作用的角度锁定刚度**。设第 $k$ 个配位分子（溶剂或阴离子）的质心/配位中心相对中心 $\text{Li}^+$ 的径向单位矢量为：
$$
\hat{\mathbf{r}}_{\text{Li}\to k} = \frac{\mathbf{r}_k - \mathbf{r}_{\text{Li}}}{\|\mathbf{r}_k - \mathbf{r}_{\text{Li}}\|} \in 1o
$$
设配体分子的局部电偶极矩矢量为 $\boldsymbol{\mu}_k = \sum_{a \in k} q_a (\mathbf{r}_a - \mathbf{r}_k^{\text{COM}}) \in 1o$（采用物理惯例：从负电荷中心指向正电荷中心）。当溶剂负电端（如羰基氧 $\delta^-$）朝向 $\text{Li}^+$、正电端背离 $\text{Li}^+$ 时，$\boldsymbol{\mu}_k$ 与 $\hat{\mathbf{r}}_{\text{Li}\to k}$ 同向平行。由此定义严格 $O(3)$ 标量不变量（两个奇宇称极矢量 $1o \otimes 1o \to 0e$ 的内积）：
$$
\cos\theta_k = \frac{\boldsymbol{\mu}_k \cdot \hat{\mathbf{r}}_{\text{Li}\to k}}{\|\boldsymbol{\mu}_k\| + \epsilon} \in [-1, 1]
$$
在强极性碳酸酯 SSIP 壳中，$\cos\theta_k \to 1$ 且壳内角度涨落极小；而在氟代醚稀释剂、空间位阻溶剂或阴离子侵入形成 CIP/AGG 时，局部静电排斥迫使配体偏转，$\langle \cos\theta_k \rangle$ 下降且角度方差增大，直接指示去溶剂化势垒软化。

### 3.4 周期性边界条件（PBC）与最小镜像约定（MIC）
无论是体相液态电解液立方模拟盒（[#2, #7, #9, #11]）还是三斜/正交电极与固态电解质晶胞（[#12–#21]），原子间真实相对位移矢量均必须通过晶格基矢矩阵 $\mathbf{L} = [\mathbf{a}, \mathbf{b}, \mathbf{c}]^\top \in \mathbb{R}^{3\times 3}$ 及晶胞平移整数向量 $\mathbf{n}_{ij} \in \mathbb{Z}^3$ 计算：
$$
\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i + \mathbf{n}_{ij}\mathbf{L}
$$
在小晶胞固态电解质中，单个邻域截断球 $r_{\text{cut}}$（如 $5.0\text{--}6.0\,\text{\AA}$）可能包含同一原子的多个周期镜像，因此必须采用**多镜像周期图（Multi-edge Periodic Graph）**而非简单丢弃次近镜像的朴素立方盒折叠，从而确保跨晶胞边界的溶剂化壳完整闭合且离子渗流通道连续无断点。

### 3.5 组分置换不变性与角色条件化（Role-Conditioned Permutation Invariance）
对齐 `MolSets` [#37]、`DiffMix` [#35] 与 `Rittig & Mitsos` [#36]，多组分液体配方编码器必须满足：
1. **置换不变性**：交换配方列表中任意两个溶剂或添加剂的输入次序，混合物潜表示严格不变；
2. **零浓度连续性**：当某组分的摩尔分数 $x_k \to 0$ 时，该组分对混合表征的贡献平滑趋于零，杜绝将“含 $0\%$ 添加剂的三元列表”与“二元列表”映射为不同特征的伪影；
3. **化学角色显式解耦**：同一分子或离子在不同浓度区间可扮演不同物理角色（主锂盐阳离子/阴离子、主溶剂、弱配位共溶剂、非溶剂稀释剂、成膜添加剂），需将组分角色嵌入与摩尔浓度共同注入集合池化权重。

---

## 4. 创新性与学术空白带分析（Creativity & Whitespace Analysis）

纵观上述 2020–2026 年的 40 篇代表性文献，当前 AI 驱动电池电解液与电极研究版图中存在一个极为清晰的**“三模态断层（Tri-Modal Chasm）”**：

```
  [Cluster D: 标量配方模型]                     [Cluster B & A-MLFF: 等变原子势/MD]
  MolSets / DiffMix / Uni-ELF                  BAMBOO / MACE-MP-0 / CHGNet / SevenNet
  ✓ 毫秒级推理、支持连续摩尔比                   ✓ 严格 O(3) 等变、保留三维配位与晶格几何
  ✗ 仅输出 l=0 标量 (σ, η, CE, RDF)            ✗ 需数万步 MD 或多镜像 NEB 才能提取脱壳/迁移机制
  ✗ 完全丢失 3D 配位不对称与脱壳方向              ✗ 无自然语言条件查询与机理符号化解释能力
                     \                               /
                      \                             /
                       ▼                           ▼
         ★ 空白核心带：YLM 文本条件化 IrrepsToText 方向查询
            + O(3)-协变 3D 去溶剂化逃逸 / 笼不对称 / 通道矢量单步读出 ★
                                    ▲
                                    |
                    [Cluster C: 材料与电池大语言模型]
                    BatteryBERT / CrystaLLM / Crystal-Text-LLM / ChatBattery
                    ✓ 支持自然语言物理假设推理与多任务条件指令
                    ✗ 将 3D 结构压扁为 1D CIF/SMILES 字符串，彻底破坏 O(3) 等变性
                    ✗ 无法感知配体偶极取向 cosθ，更无法输出旋转协变的 3D 物理矢量
```

### 为什么“文本条件化 `IrrepsToText` 方向性查询 + $O(3)$-协变 3D 去溶剂化逃逸/笼不对称矢量读出”具有关键桥梁价值？

1. **跨越“标量统计平均”与“三维局域各向异性机制”的鸿沟**：
   在真实电池界面处，$\text{Li}^+$ 的脱壳与嵌入本质上是一个**高度方向性的对称性破缺过程**——在完美对称的四配位 SSIP 笼中，配体偶极径向抵消，笼偶极不对称矢量近似为零，脱壳需克服极高的全向能垒；而当一个 $\text{FSI}^-$ 阴离子或弱溶剂化氟代醚分子进入第一配位壳形成非对称 CIP 时，溶剂化笼立刻产生显著的 **$l=1$ 极矢量偶极不对称（Cage Dipole-Asymmetry Vector）** 与 **$l=2$ 四极形变（Quadrupole Distortion）**，为 $\text{Li}^+$ 打开特定的低阻三维逃逸鞍点开口。`CALiSol-23`、`MolSets` 乃至 `Uni-ELF` 的球面平均 RDF 均将这一决定快充与低温动力学成败的三维方向特征完全抹平；而 YLM 通过保留 `0e + 1o + 2e` 不可约表示并直接读出 $O(3)$-协变 3D 逃逸矢量 $\mathbf{v}_{\text{escape}}$，首次在毫秒级单次前向传播中捕获了这种局域对称性破缺。
2. **用单步文本条件化几何探测替代昂贵的轨迹后处理与多镜像 NEB 搜索**：
   在传统工作流（如 `BAMBOO` MD + `SolvationAnalysis` 或 `CHGNet/MACE` + `LiTraj` NEB）中，研究者若想回答不同工况条件下的方向性问题（例如：*“在低温快充高过电位下，当前 CIP 溶剂化笼的 $\text{Li}^+$ 优先从哪一侧配体间隙脱壳？”* 或 *“在该正极晶胞的特定空位处，最低势垒跃迁指向哪个邻位通道？”*），必须针对每种边界条件重新设置约束 MD 或伞形采样/NEB。通过 `IrrepsToText` 跨模态桥，自然语言文本探针可以直接调制高阶不可约特征通道的注意力门控，使同一个等变几何主干根据文本任务指令按需激活对应的 $1o$ 极矢量（特定有向脱壳/单步跳跃矢量）或 $2e$ 张量（中心对称晶格无向扩散主轴），并在生成可解释物理文本分析的同时输出严格满足 $\mathbf{v}(R\cdot\mathcal{X}) = R\,\mathbf{v}(\mathcal{X})$ 的三维空间矢量。
3. **打通宏观多组分配方层与微观三维配位结构层的双向闭环**：
   将置换不变的混合配方先验（摩尔比、组分角色、温度 VTF 标度）作为上下文条件注入局域溶剂化笼的等变不可约表示，既赋予配方模型微观三维几何可解释性，又避免了在大体积显式溶液盒上运行全原子 AIMD 的指数级算力开销。

---

## 5. 面向 YLM 电解液模块（`src/toolkit/domains.py`）的 5 项具体架构升级建议

基于上述 40 篇文献提炼的物理定律与几何约束，建议在 YLM 电解液领域工具包（`src/toolkit/domains.py`）中集成以下 **5 项高优先级模块升级**（均严格保持 $O(3)$ 群等变性与可微性）：

### 升级 1：引入溶剂化笼偶极不对称矢量与去溶剂化逃逸矢量算子（`CageAsymmetry & DesolvationEscapeReadout`, $1o \oplus 2e$）
- **文献依据**：Holoubek et al. (*Nat. Energy* 2021 [#3])、Hou & Persson (*Chem. Sci.* 2021 [#2])、`OMol25` (2025 [#10])。
- **技术实现**：围绕中心阳离子 $\text{Li}^+$，利用平滑径向截断权重 $w(r_{\text{Li},j})$ 同步构造两类物理协变几何序参量：
  1. **空间配位空腔偏移极矢量（Steric Cavity Escape Vector, $1o$）**：
     $$\mathbf{v}_{\text{cavity}} = -\sum_{j \in \mathcal{N}(\text{Li})} w(r_{\text{Li},j})\,\hat{\mathbf{r}}_{\text{Li}\to j}$$
     当配位壳完备对称（如正四面体/正八面体）时 $\|\mathbf{v}_{\text{cavity}}\| = 0$；当某侧配体松动或出现空间缺口时，$\mathbf{v}_{\text{cavity}}$ 自动指向空间位阻最小的物理脱壳开口方向。
  2. **静电加权笼偶极不对称矢量（Electrostatic Cage-Asymmetry Vector, $1o$）与四极张量（$2e$）**：
     结合配位原子的局部部分电荷/ESP 权重 $\tilde{q}_j$ 计算 $\mathbf{v}_{\text{elec}} = \sum_{j \in \mathcal{N}(\text{Li})} w(r_{\text{Li},j})\,\tilde{q}_j\,\hat{\mathbf{r}}_{\text{Li}\to j}$ 及无迹对称二阶张量 $\mathbf{Q}_{\text{cage}} = \sum_j w(r_{\text{Li},j})\left(\hat{\mathbf{r}}_{\text{Li}\to j}\otimes\hat{\mathbf{r}}_{\text{Li}\to j} - \frac{1}{3}\mathbf{I}\right) \in 2e$，共同馈入等变读出头预测 3D 去溶剂化逃逸方向矢量 $\hat{\mathbf{v}}_{\text{escape}} \in 1o$。

### 升级 2：显式编码配体偶极–锂离子径向对齐度特征（`LigandDipoleAlignment`, $\cos\theta \in 0e$）
- **文献依据**：Holoubek et al. (*Nat. Energy* 2021 [#3])、Wu et al. (*Adv. Energy Mater.* 2023 [#6])。
- **技术实现**：在 `domains.py` 溶剂化团簇图构建阶段，为第一配位壳内每个配体分子 $k$ 计算其分子级电偶极矢量 $\boldsymbol{\mu}_k \in 1o$（由 Gasteiger/Mulliken 电荷或等变原子偶极子求和得到），并提取其与 $\text{Li}^+\to k$ 径向矢量的内积标量特征：
  $$s_k^{\text{align}} = \frac{\boldsymbol{\mu}_k \cdot \hat{\mathbf{r}}_{\text{Li}\to k}}{\|\boldsymbol{\mu}_k\| + \epsilon} \in 0e, \qquad \boldsymbol{\tau}_k^{\text{torque}} = \hat{\boldsymbol{\mu}}_k \times \hat{\mathbf{r}}_{\text{Li}\to k} \in 1e\text{ (赝矢量)}$$
  同时统计壳层平均径向锁定度 $\bar{s}_{\text{align}}$ 与壳内离散度 $\sigma_{\text{align}}^2$，直接量化“偶极锁定刚度（dipole-locking stiffness）”并注入 `IrrepsToText` 的物理摘要槽位，使语言输出能定量区分强锁定碳酸酯壳与弱配位醚/阴离子壳。

### 升级 3：可微连续 SSIP / CIP / AGG 软拓扑物种分类器（`SoftSpeciationClassifier`, $\Delta^2 \subset 0e$）
- **文献依据**：Cohen & Persson (`SolvationAnalysis`, 2023 [#7])、Kim & Cui (*PNAS* 2023 [#5])、`Uni-ELF` (2024 [#38])。
- **技术实现**：在 `domains.py` 中实现无需硬阈值切片的可微连续物种判别算子：记平滑阴离子配位数 $\tilde{N}_A(i) = \sum_{m \in \text{Anions}} \sigma_{\text{soft}}(d_{\min}(i,m) - r_c)$ 及配位阴离子的过剩阳离子桥接数 $\tilde{B}_{\text{agg}}(i) = \sum_{m \in \text{Anions}} A_{im}^{\text{soft}} \cdot \text{ReLU}\!\left(\sum_{i'} A_{i'm}^{\text{soft}} - 1\right)$，映射为连续归一化三元概率分布 $(p_{\text{SSIP}}, p_{\text{CIP}}, p_{\text{AGG}})$。该概率向量既可作为辅助自监督任务训练等变编码器，又能作为硬物理接地标签注入 `IrrepsToText` 提示前缀，消除大模型对配位态归属的语言幻觉。

### 升级 4：PBC 最小镜像多边图与“有向跃迁 $1o$ + 中心对称通道 $2e$”双模晶体迁移读出（`PBCMigrationChannelReadout`）
- **文献依据**：He & Shi (`BVSE/CAVD`, 2020 [#13])、`CHGNet` (2023 [#15])、`MACE-MP-0` (2024 [#16])、`LiTraj` (Dembitskiy et al., 2025 [#21])。
- **技术实现**：
  1. 在 `domains.py` 的周期性电极/固态电解质图接口中强制采用携带晶胞平移整数 $\mathbf{n}_{ij} \in \mathbb{Z}^3$ 的精确最小镜像边位移 $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i + \mathbf{n}_{ij}\mathbf{L}$；
  2. **解决中心对称晶体的极矢量抵消陷阱**：在具有反演对称中心（如橄榄石相 $\text{LiFePO}_4$ $Pnma$ 或石榴石型 LLZO）的完整晶胞中，全局平均的 $1o$ 奇宇称极矢量必然恒等于 $\mathbf{0}$！因此，`domains.py` 应实现双模协变读出：
     - 当文本探针指定**局域空位跃迁对 $(i \to j)$** 时，读出局域鞍点初猜跃迁极矢量 $\mathbf{v}_{\text{hop}}^{(i\to j)} \in 1o$；
     - 当文本探针询问**晶胞全局快速导锂通道取向**（如 $\text{LiFePO}_4$ 沿 $[010]$ $b$ 轴的一维通道）时，读出晶胞池化后的**无迹对称二阶扩散各向异性张量 $\mathbf{D}_{\text{channel}} \in 2e$**，并取其最大特征值对应的主特征向量 $\pm\hat{\mathbf{u}}_{\max}$ 作为无向导锂通道轴线！这一设计在群论上彻底消除了反演对称晶体的符号简并退化。

### 升级 5：满足零浓度极限的角色感知混合配方聚合器与可微 VTF 温度标度头（`RoleConditionedMixtureAggregator`）
- **文献依据**：Bradford & Shao-Horn (*ACS Cent. Sci.* 2023 [#33])、`DiffMix` (*Nat. Commun.* 2024 [#35])、`MolSets` (*PRX Energy* 2024 [#37])、Yang et al. (*Nat. Mach. Intell.* 2025/2026 [#40])。
- **技术实现**：在 `domains.py` 的配方级封装中，将各组分分子/离子的等变池化不变量 $\mathbf{z}_i \in 0e$ 与化学角色嵌入 $\mathbf{e}_{\tau_i}$（区分 `Li_salt_cation`、`salt_anion`、`primary_solvent`、`diluent`、`additive`）拼接后，严格采用摩尔分数 $x_i$ 线性门控的自注意力聚合：
  $$\mathbf{h}_{\text{mix}} = \sum_{i=1}^{K} x_i \, \text{MLP}\!\left(\mathbf{z}_i \,\|\, \mathbf{e}_{\tau_i} \,\|\, \sum_{j=1}^{K} x_j \alpha_{ij} \mathbf{v}_j\right)$$
  从而在数学上严格保证集合置换不变性与 $\lim_{x_k \to 0} \mathbf{h}_{\text{mix}} = \mathbf{h}_{\text{mix}\setminus\{k\}}$ 零浓度连续性；同时在宏观电导率分支配备物理约束的 VTF 参数预测头 $\ln\sigma(T) = \ln A(\mathbf{h}_{\text{mix}}) - \frac{B(\mathbf{h}_{\text{mix}})}{T - T_0(\mathbf{h}_{\text{mix}})}$（约束 $B > 0,\; T_0 < T_{\min}$），使宏观配方电导率预测与局域 3D 溶剂化笼的 `IrrepsToText` 方向性推理形成端到端多尺度闭环。
