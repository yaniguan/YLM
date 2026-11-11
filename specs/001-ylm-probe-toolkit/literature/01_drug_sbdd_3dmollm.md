# 3D 基于结构的药物设计（SBDD）、片段生长/连接子出射向量（Exit-Vector）、3D 药效团与 3D 分子大语言模型文献综述（2020–2026）

> **调研范围与验证说明**：本报告系统梳理并核验了 **2020–2026 年间发表于顶级期刊/会议（Nature, Science, Cell, Nat. Methods, Nat. Mach. Intell., Nat. Comput. Sci., Nat. Commun., Chem. Sci., JCIM, ICML, NeurIPS, ICLR, ACL, EMNLP, AAAI）及权威预印本（arXiv / bioRxiv）的 40 篇真实代表性论文**，覆盖四大核心技术方向（Cluster A–D），旨在为 **YLM 药物先导化合物优化与药效团分析模块（`src/toolkit/domains.py`）** 提供物理标准对齐（Alignment）、创新空白定位（Creativity & Whitespace）以及可落地的几何算法升级方案。

---

## 一、 40 篇核心文献全景对照表（2020–2026）

表中对每篇工作的核心张量/模态表征严格按对称性阶数标注：
- **`0e`（偶宇称标量）**：距离矩阵、原子类型、电荷、能量或不变图特征；
- **`1o`（奇宇称极向量）**：笛卡尔坐标偏移 $\Delta \mathbf{r}$、偶极矩、力、键轴与出射向量（在正交变换 $Q \in O(3)$ 下满足 $\mathbf{v} \mapsto Q\mathbf{v}$，反演下变号）；
- **`1e`（偶宇称赝向量/轴向量）** / **`2e`（二阶无迹对称偶张量）**：环平面法向交叉积（`1e`）或芳环四极/平面投影张量 $\mathbf{n}\otimes\mathbf{n} - \frac{1}{3}\mathbf{I}$（`2e`，天然消除法向 $\pm\mathbf{n}$ 符号歧义）；
- **`irreps`**：基于球谐函数 $Y_\ell^m$ 与 Clebsch–Gordan 张量积的高阶不可约表征（如 `e3nn`）；
- **`voxels`**：三维体素密度网格；
- **`text`**：1D 序列/自然语言/量化坐标词元。

### Cluster A：3D 等变口袋条件分子生成与分子对接（11 篇，#1–#11）

| No. | 论文标题、作者与年份/发表载体 | 核心表征 (`0e` / `1o` / `irreps` / `voxels`) | 方向性与几何等变处理机制 | 核心局限性 |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Pocket2Mol**: *Pocket2Mol: Efficient Molecular Sampling Based on 3D Protein Pockets* (Peng et al., **ICML 2022**) | `0e` 标量 + `1o` 极向量（GVP 几何向量感知机） | 基于 $E(3)$-等变 GVP 消息传递同时更新口袋-配体 `0e` 特征与 `1o` 空间方向向量，自回归预测前沿原子与连续高斯混合位置分布 | 自回归逐原子采样存在曝光偏差与顺序敏感性；无自然语言交互能力，无法按语义查询特定生长方向 |
| **2** | **EquiBind**: *EquiBind: Geometric Deep Learning for Drug Binding Structure Prediction* (Stärk et al., **ICML 2022**) | `0e` 节点特征 + `1o` 3D 坐标（IEGMN） | 利用 $SE(3)$-等变图匹配网络（IEGMN）结合交叉注意力关键点对齐（Kabsch 刚体变换），一步直接回归配体结合构象 | 单次确定性坐标均值回归在多模态对称口袋中易产生键长畸变与严重蛋白-配体空间碰撞（steric clashes） |
| **3** | **TANKBind**: *TANKBind: Trigonometry-Aware Neural NetworKs for Drug-Protein Binding Structure Prediction* (Lu et al., **NeurIPS 2022**) | `0e` 蛋白-配体成对距离矩阵与三角不等式注意力 | 完全工作于 $O(3)$-不变标量距离空间，通过三角自洽模块预测配体-功能块距离矩阵后再经梯度下降重构 3D 坐标 | 丢失显式手性与反演奇偶区分（距离矩阵对对映异构反射不变），且后处理坐标重构耗时并可能陷入局部极小 |
| **4** | **TargetDiff**: *3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction* (Guan et al., **ICLR 2023**) | `0e` 离散原子类别概率 + `1o` 连续 3D 坐标（EGNN） | 在零质心（CoM-free）子空间构建 $SE(3)$-等变扩散过程，利用径向相对位移 $(\mathbf{x}_i - \mathbf{x}_j)$ 加权更新配体去噪漂移向量 | EGNN 仅传递成对径向 `1o` 分量而缺少高阶角动量（`2e` 平面度/扭转角），常生成非平面稠环与高张力不合理构象 |
| **5** | **DiffDock**: *DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking* (Corso et al., **ICLR 2023**) | `irreps`（`e3nn` 球谐张量积 `0e, 0o, 1o, 1e`） | 在配体自由度积流形 $\mathbb{R}^3 \times SO(3) \times \mathbb{T}^m$（平移、整体旋转、可旋转键二面角）上定义等变分数网络与独立置信度评估器 | 假设蛋白口袋刚性（除非额外扩展侧链）；仅输出对接构象分布，不支持化学修饰或语义级结构-活性关系（SAR）问答 |
| **6** | **DecompDiff**: *DecompDiff: Diffusion Models with Decomposed Priors for Structure-Based Drug Design* (Guan et al., **ICML 2023**) | `0e` 原子/键特征 + `1o` 3D 坐标 + 子口袋高斯先验 | 将配体显式分解为结合子口袋的“臂（Arms）”与连接中心的“骨架（Scaffold）”，从各向异性高斯先验启动 $SE(3)$-等变扩散 | 依赖启发式子口袋聚类/先验中心提取（如 AlphaSpace2）；缺乏语义条件控制来指定保留/替换哪一条支链臂 |
| **7** | **DiffSBDD**: *Structure-based drug design with equivariant diffusion models* (Schneuing et al., **Nat. Comput. Sci. 2024**; preprint 2022) | `0e` 标量节点特征 + `1o` 3D 坐标（EGNN） | 支持口袋条件（pocket-conditioned）与联合绘画（inpainting）两种 $E(3)$-等变扩散模式，可通过遮罩固定母核片段进行局部衍生化 | 基于 EGNN 的成对距离更新对芳环共面性与杂原子孤对电子圆锥角约束较弱，需依赖外部力场松弛修正 |
| **8** | **MolCRAFT**: *MolCRAFT: Structure-Based Drug Design in Continuous Parameter Space* (Qu et al., **ICML 2024**) | `0e` 连续概率单纯形参数 + `1o` 3D 坐标（$SE(3)$ BFN） | 采用贝叶斯流网络（Bayesian Flow Networks）在全连续参数空间联合更新原子类型与 $SE(3)$-等变坐标，显著降低离散-连续噪声失配导致的几何碰撞 | 仍属于纯无语言条件生成模型，无法接受药物化学家关于极性方向、氢键锚点或代谢不稳定位点的文本指令 |
| **9** | **VoxBind**: *Structure-based drug design by denoising voxel grids* (Pinheiro et al., **ICML 2024**) | 3D 多通道原子密度网格 `voxels` | 放弃点云等变消息传递，改用随机旋转数据增强的 3D U-Net 在 $0.25\text{ \AA}$ 体素网格上进行 Walk-Jump Langevin 采样 | 严格 $O(3)$ 等变性被网格离散化破坏（仅有近似旋转鲁棒性），内存开销随盒子体积立方增长，无法直接读出连续解析向量 |
| **10** | **SurfDock**: *SurfDock is a surface-informed diffusion generative model for reliable and accurate protein–ligand complex prediction* (Cao et al., **Nat. Methods 2025**) | 蛋白分子表面网格（`0e` 静电/疏水标量 + 表面法向量）+ `irreps` 扩散 | 将蛋白口袋编码为兼具几何曲率、法向量与静电势的分子表面图，结合流形扩散大幅降低 PoseBusters 物理碰撞率 | 专注于给定化学式的复合物构象预测，不直接输出可替换 C–H 出射向量排序或自然语言优化建议 |
| **11** | **PocketXMol**: *Unified modeling of 3D molecular generation via atomic interactions with PocketXMol* (Peng et al., **Cell 2026**) | `0e` + `1o` 统一全原子交互去噪张量 | 在原子级统一建模小分子对接、从头设计、连接子设计与多肽结合，通过任务掩码与 $SE(3)$-等变去噪器共享底层物理交互先验 | 控制接口限定于结构掩码与离散任务条件标签，缺乏开放式自然语言多步推理与显式物理向量（如偶极/出射向量）符号化读出 |

---

### Cluster B：3D 片段生长、连接子设计（Linker Design）与出射向量条件生成（10 篇，#12–#21）

| No. | 论文标题、作者与年份/发表载体 | 核心表征 (`0e` / `1o` / `voxels` / `text`) | 方向性与出射向量（Exit-Vector）处理机制 | 核心局限性 |
| :--- | :--- | :--- | :--- | :--- |
| **12** | **DeLinker**: *Deep Generative Models for 3D Linker Design* (Imrie et al., **J. Chem. Inf. Model. 2020**) | 2D 图（`0e`）+ 锚点距离 $d$ 与出射向量相对夹角标量编码 | 将两个片段锚点的出射键向量压缩为 $O(3)$-不变标量距离与相对余弦角特征，条件驱动 2D GNN 逐键生成连接子再筛选 3D 构象 | 生成阶段发生在 2D 拓扑图上，忽略蛋白口袋全原子空间排斥，常产生在 3D 口袋中严重碰撞的连接子 |
| **13** | **DEVELOP**: *Deep Generative Design with 3D Pharmacophoric Constraints* (Imrie et al., **Chem. Sci. 2021**) | 2D GNN（`0e`）+ 3D 药效团体素 `voxels`（3D CNN） | 在 DeLinker 出射向量距离/角度标量约束基础上，引入 3D CNN 提取连接区域药效团体素网格（芳环/氢键供受体）指导 R 基团生长与连接 | 3D CNN 体素特征不具备精确 $O(3)$ 等变性（依赖坐标系对齐），且输出的仍是 2D 分子图而非精确 3D 坐标 |
| **14** | **DeepFrag**: *DeepFrag: a deep convolutional neural network for fragment-based lead optimization* (Green, Koes, & Durrant, **Chem. Sci. 2021**) | 受体-配体-生长点 3D 体素 `voxels` $\to$ 2D 片段指纹（`0e`） | 以母核连接原子为中心裁剪 3D 体素网格并经旋转增强送入 3D CNN，预测最优修饰取代基指纹并在片段库中检索最近邻 | 将连续 3D 出射方向坍缩为封闭库指纹分类问题，不显式计算或验证 C–H 出射射线的立体间隙（clearance） |
| **15** | **3DLinker**: *3DLinker: An E(3) Equivariant Variational Autoencoder for Molecular Linker Design* (Huang et al., **ICML 2022**) | `0e` 节点标量 + `1o` 向量消息传递（$E(3)$-等变 VAE） | 无需预先手工指定锚点出射向量，利用 $E(3)$-等变图自编码器联合预测锚点原子、连接子拓扑与精确 3D 原子坐标 | 仅面向双片段真空几何条件设计，未显式融入口袋蛋白空间位阻场与静电互补性 |
| **16** | **FragDiff**: *Pocket-specific 3D Molecule Generation by Fragment-based Autoregressive Diffusion Models* (Peng et al., **OpenReview `HGsoe1wmRW5` 2023**) | `0e` 标量 + `1o` 向量（GVP + 局部片段 $E(3)$ 扩散） | 在蛋白口袋内以化学片段为单位自回归生长：每一步先确定母核出射连接位点，再用等变扩散生成下一个局部 3D 片段的原子与键 | 片段级自回归一旦早期母核取向偏离最优氢键锚点则后续误差累积，不支持交互式文本提问与多候选出射向量对比解释 |
| **17** | **D3FG**: *Functional-Group-Based Diffusion for Pocket-Specific Molecule Generation and Elaboration* (Lin et al., **NeurIPS 2023**) | 刚性官能团（`1o` 质心 + $SO(3)$ 旋转标架）+ 质点连接原子 | 将芳环、酰胺等常见药效官能团建模为带局部正交标架的刚体（完美保持环平面性），再通过等变扩散协同生成单原子连接桥 | 官能团词表固定（无法覆盖罕见杂环或非经典生物电子等排体），缺乏自然语言驱动的官能团定向替换接口 |
| **18** | **LinkerNet**: *LinkerNet: Fragment Poses and Linker Co-Design with 3D Equivariant Diffusion* (Guan et al., **NeurIPS 2023 Spotlight**) | 片段刚体标架 $(\mathbf{R} \in SO(3), \mathbf{t} \in \mathbb{R}^3)$ + 原子级 `0e`/`1o` 扩散 | 针对 PROTAC 等片段相对朝向未知的场景，在牛顿-欧拉力学启发下同步扩散两侧弹头片段的 $SE(3)$ 姿态（自动调节出射向量夹角）与连接子 | 主要优化真空或简化界面下的片段-连接子协同几何，未建立三元复合物全蛋白语义推理能力 |
| **19** | **DiffLinker**: *Equivariant 3D-conditional diffusion model for molecular linker design* (Igashov et al., **Nat. Mach. Intell. 2024**) | `0e` 原子类型 + `1o` 3D 坐标（$E(3)$-等变 EGNN） | 突破仅能连接两个片段的限制，支持在显式蛋白口袋点云排斥条件下自动预测连接子原子数并同时桥接任意数量（$\ge 2$）的 3D 片段 | 生成过程为黑盒扩散采样，不提供各潜在锚点 C–H 出射向量的位阻裕度分解或方向性物理归因 |
| **20** | **SyntheMol**: *Generative AI for designing and validating easily synthesizable and structurally novel antibiotics* (Swanson et al., **Nat. Mach. Intell. 2024**) | 2D 合成砌块与反应模板图（`0e` 标量属性评分 + MCTS） | 通过蒙特卡洛树搜索在数百亿组合化学反应空间中拼接商业可得砌块，确保 100% 实验室高可合成性并通过湿实验验证抗菌活性 | 完全缺失 3D 靶标口袋几何、药效团方向与立体出射向量表征，无法解决靶向特定结合构象的结构优化问题 |
| **21** | **TamGen**: *TamGen: drug design with target-aware molecule generation through a chemical language model* (Wu et al., **Nat. Commun. 2024**) | 口袋坐标感知 Transformer（`0e` 距离偏置注意力）$\to$ 1D SMILES `text` | 用几何感知编码器提取蛋白口袋结合位点表征，通过交叉注意力输入类 GPT 化学语言模型，支持从头生成与基于母核片段的先导化合物优化 | 解码端仅输出 1D SMILES 字符串，完全丢失生成分子在口袋中的 3D 结合姿态与取代基空间出射方向向量 |

---

### Cluster C：3D 药效团、静电势/形状/向量场对齐与全原子相互作用模型（9 篇，#22–#30）

| No. | 论文标题、作者与年份/发表载体 | 核心表征 (`0e` / `1o` / `irreps` / `voxels`) | 方向性、静电场与药效团对齐机制 | 核心局限性 |
| :--- | :--- | :--- | :--- | :--- |
| **22** | **ESP-Sim**: *On the Value of Using 3D Shape and Electrostatic Similarities in Deep Generative Methods* (Bolcato, Heid, & Boström, **J. Chem. Inf. Model. 2022**) | 3D 高斯体积重叠 + 库仑静电势（ESP）空间标量场（`0e`） | 在分子范德华表面外径向网格积分三维形状与量子化学/部分电荷静电势相似度（Carbo/Hodgkin 指数），作为强化学习奖励引导生物电子等排跃迁 | 需要预先将候选构象刚体叠合到参考配体坐标系；作为标量奖励函数无法直接端到端输出等变梯度或方向向量 |
| **23** | **E3Bind**: *E3Bind: An End-to-End Equivariant Network for Protein-Ligand Docking* (Zhang et al., **ICLR 2023**) | `0e` 成对交互张量 + `1o` $E(3)$-等变坐标迭代更新 | 结合局部口袋图上下文与三角几何距离自洽约束，通过端到端 $E(3)$-等变消息传递迭代修正柔性配体结合坐标 | 无显式静电势或方向性药效团（如芳环法向、孤对电子圆锥）建模，局部极性氢键几何精度受限 |
| **24** | **PGMG**: *A pharmacophore-guided deep learning approach for bioactive molecular generation* (Zhu et al., **Nat. Commun. 2023**) | 药效团完全图（`0e` 最短路径/欧氏成对距离权重）$\to$ `text` SMILES | 将 3D 药效团假设编码为节点类型 + 成对欧氏距离完全图，引入潜变量与 Transformer 生成满足多对多药效团映射的 SMILES | 药效团仅以点坐标间的标量距离表示（缺失氢键圆锥轴向与芳环平面法向 `1o`/`2e` 约束），且仅输出 1D SMILES |
| **25** | **IPDiff**: *Protein-Ligand Interaction Prior for Binding-aware 3D Molecule Diffusion Models* (Huang et al., **ICLR 2024**) | `0e` + `1o` 亲和力监督交互网络（IPNet）+ 先验偏移扩散 | 预训练蛋白-配体 3D 物理相互作用先验网络，在前向与反向扩散轨迹中注入亲和力驱动的 `1o` 坐标先验偏移（prior-shifting） | 交互特征作为隐式潜向量注入去噪器，无法显式提取可解释的药效团向量或回答特定基团的极性互补问题 |
| **26** | **PharmacoNet**: *PharmacoNet: Deep Learning-Guided Pharmacophore Modeling for Ultra-Large-Scale Virtual Screening* (Seo & Kim, **Chem. Sci. 2024**; NeurIPS 2023 W) | 3D 口袋体素 `voxels`（实例分割）$\to$ 粗粒化距离图（`0e`） | 利用 3D U-Net 在空蛋白口袋内直接分割识别 7 类药效团热点区域（HBD/HBA/芳环/疏水/正负电荷/卤键），通过图匹配实现百万级超快虚拟筛选 | 口袋热点提取后简化为球形中心与距离容差匹配，弱化了芳环 $\pi$-堆积倾角与氢键供受体矢量共线性判据 |
| **27** | **RoseTTAFold All-Atom (RFAA)**: *Generalized biomolecular modeling and design with RoseTTAFold All-Atom* (Krishna et al., **Science 2024**) | 1D/2D `0e` 轨迹 + 残基刚体标架 + 全原子 `irreps`（`0e`+`1o` SE(3)-Transformer） | 将生物大分子残基标架与小分子/金属离子显式原子图融合，注入手性二面角符号特征以突破 $E(3)$ 反演歧义并精确建模共价/非共价配位 | 参数规模与多序列比对（MSA）推理开销巨大；仅作为结构预测/蛋白设计基座，无自然语言分析或先导优化建议能力 |
| **28** | **Chai-1**: *Chai-1: Decoding the molecular interactions of life* (Chai Discovery / Boitreaud et al., **bioRxiv 2024**) | 蛋白语言模型（pLM）单序列 `0e` + 成对特征（支持实验约束）+ 原始 `1o` 3D 坐标扩散 | 无需深度 MSA 即可高精度预测蛋白-小分子复合物，支持注入口袋接触/距离实验约束，采用随机旋转增强的非等变 Transformer 直接去噪原子坐标 | 放弃架构级严格 $O(3)$ 等变性（依赖训练期旋转增强，推理时对输入旋转存在微小数值漂移），且不支持语义交互问答 |
| **29** | **Boltz-1 / Boltz-2**: *Boltz-1: Democratizing Biomolecular Interaction Modeling* (Wohlwend et al., **bioRxiv 2024**) & *Boltz-2* (Passaro et al., **bioRxiv 2025**) | 全开源全原子 `0e` Pairformer 主干 + 数据增强 `1o` 3D 扩散 + 双标量亲和力头（Boltz-2） | 开源复现并超越 AF3 级蛋白-配体全原子复合物结构预测，Boltz-2 进一步耦合结合可能性与 $\Delta G / \text{IC}_{50}$ 标量亲和力预测，速度比 FEP 快千倍 | 亲和力头仅输出全局 `0e` 标量结合自由能，不分解单个位点的立体出射向量净空度或局部偶极/药效团方向贡献 |
| **30** | **ShEPhERD**: *ShEPhERD: Diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design* (Adams et al., **ICLR 2025 Oral**; arXiv:2411.04130) | `e3nn` 高阶 `irreps`（`0e` 静电势 + 表面点云 + `1o` 药效团方向向量） | **代表性里程碑**：利用 `e3nn` 球谐张量积联合扩散 3D 分子结构、分子表面形状、静电势（ESP）与**显式带方向单位向量的药效团**（并对芳环正反法向 $\pm\mathbf{v}$ 设计对称容忍机制） | 纯几何扩散生成模型，无大语言模型接口，无法接受自然语言条件查询，也无法针对输入复合物开展可解释的交互式先导优化诊断 |

---

### Cluster D：3D 分子与药物大语言模型 / 多模态分子-文本模型（10 篇，#31–#40）

| No. | 论文标题、作者与年份/发表载体 | 核心表征 (`0e` / `text`) | 跨模态对齐与 3D 方向性处理现状 | 核心局限性（尤其是 $O(3)$ 向量坍缩问题） |
| :--- | :--- | :--- | :--- | :--- |
| **31** | **MoleculeSTM**: *Multi-modal molecule structure–text model for text-based retrieval and editing* (Liu et al., **Nat. Mach. Intell. 2023**) | 2D GNN / SMILES（`0e`）+ SciBERT `text` | 在 28 万 PubChem 结构-文本对上通过对比学习对齐全局分子图嵌入与文本嵌入，支持零样本检索与潜空间分子编辑 | 纯 2D 拓扑 + 全局 `0e` 标量池化，完全缺失 3D 构象、蛋白口袋以及任何空间方向向量信息 |
| **32** | **DrugChat**: *DrugChat: Towards Enabling ChatGPT-Like Capabilities on Drug Molecule Graphs* (Liang et al., **arXiv:2305.03903, 2023**) | 2D GNN 图特征（`0e`）+ 线性投影层 $\to$ Vicuna `text` | 仿照 LLaVA 架构，将 GNN 提取的分子图全局向量经线性适配器映射为单个软提示词元输入 LLM 进行多轮药物属性问答 | 分子被压缩为单一 $O(3)$-不变标量 token，无原子级 3D 坐标感知，更无法输出空间方向向量 |
| **33** | **MolCA**: *MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter* (Liu et al., **EMNLP 2023**) | 2D GINE 图（`0e`）+ SMILES $\to$ Q-Former（`0e`）$\to$ Galactica | 引入 BLIP-2 风格的跨模态投影器 Q-Former，用一组可学习标量查询向量提取分子图局部特征并喂入带 LoRA 的语言模型 | 仅编码 2D 分子图，且 Q-Former 输出固定数量的纯标量（`0e`）词元，完全不具备 3D 空间几何与方向性表达能力 |
| **34** | **Mol-Instructions**: *Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models* (Fang et al., **ICLR 2024**) | 纯 1D `text` + SELFIES / FASTA 序列词元 | 构建百万级分子、蛋白质与生物文本指令微调数据集，将化学结构统一序列化为无语法错误的 SELFIES 文本进行指令微调 | 彻底退化为 1D 字符串建模，无显式 3D 欧氏距离、口袋立体位阻或极向量表征 |
| **35** | **3D-MoLM**: *Towards 3D Molecule-Text Interpretation in Language Models* (Li et al., **ICLR 2024**) | Uni-Mol 3D 编码器（`0e` 成对距离）$\to$ **3D Q-Former（`0e` 标量 token）** $\to$ Llama-2 | 用预训练 `Uni-Mol` 提取 3D 构象特征（如 HOMO-LUMO 能隙、偶极矩大小等 3D 相关标量性质），经 Q-Former 投影进 Llama-2 并构建 `3D-MoIT` 指令集 | **典型标量瓶颈**：`Uni-Mol` 本身依赖 $SE(3)$-不变距离矩阵，且经 Q-Former 压缩为纯标量 `0e` 词元后，**3D 空间参考系与 $O(3)$ 方向向量（如偶极方向 `1o`、出射向量）被彻底抹除** |
| **36** | **nach0**: *nach0: multimodal natural and chemical languages foundation model* (Livne et al., **Chem. Sci. 2024**) | SMILES 化学词元 + 生物医学文献/专利 `text`（T5 编码-解码器） | 在海量文献、专利与化学反应式上联合预训练编码器-解码器基础模型，兼顾医药问答、逆合成预测与分子设计 | 纯 1D 化学字符串与自然语言联合词表，不接受 3D 蛋白-配体复合物坐标输入，无法推理 3D 空间构效关系 |
| **37** | **BioT5+**: *BioT5+: Towards Generalized Biological Understanding with IUPAC Integration and Multi-task Tuning* (Pei et al., **Findings of ACL 2024**) | SELFIES + IUPAC 命名 + 数值词元 + FASTA + `text`（`0e`） | 引入 IUPAC 系统命名桥梁与逐位数值分词机制，提升分子与蛋白跨模态属性预测及文本理解能力 | 无 3D 原子坐标通道，所有几何与相互作用均依赖 1D 序列隐式推断 |
| **38** | **Lingo3DMol**: *Generation of 3D molecules in pockets via a language model* (Feng et al., **Nat. Mach. Intell. 2024**) | 口袋局部坐标编码 + 片段化 `FSMILES` `text` + **离散化局部球/直角坐标 token** | 设计基于片段的 FSMILES 语法，结合非共价相互作用锚点预测与局部参考系下的量化 3D 坐标词元，在口袋中自回归生成 3D 配体 | 离散化坐标词元破坏连续 $O(3)$ 协变性（存在网格量化误差与局部标架拼接累积漂移），且仅作为专用生成器，不支持自然语言问答 |
| **39** | **BindGPT**: *BindGPT: A Scalable Framework for 3D Molecular Design via Language Modeling and Reinforcement Learning* (Zholus et al., **AAAI 2025**) | 自回归 GPT 词表：原子符号 + **$0.1\text{ \AA}$ 量化 XYZ 坐标 `text` token** | 将口袋与配体的 3D 笛卡尔坐标直接量化为离散文本词元，配合大规模预训练、随机旋转增强与强化学习微调优化亲和力 | 直接将原始坐标写成文本严重违反精确 $O(3)$ 等变性（同一复合物旋转后对应完全不同的 token 序列），且无法保证反演/旋转协变输出 |
| **40** | **3D-MolT5**: *3D-MolT5: Leveraging Discrete Structural Information for Molecule-Text Modeling* (Pei et al., **ICLR 2025**) | 离散 3D 指纹词元（`E3FP` 局部球壳子结构 `0e`）+ SELFIES + `text` | 将每个原子周围的 3D 局部构象哈希映射为离散扩展三维指纹（E3FP）词元，与文本和 1D 分子序列融合进统一 T5 模型 | E3FP 本质上是局部旋转**不变（`0e`）**的离散哈希索引，彻底丢弃了全局/分子间相对取向以及连续 `1o`/`2e` 方向张量 |

---

## 二、 领域共性规律综合提炼（Alignment：YLM 药物模块必须对齐的物理与行业标准）

跨观上述 40 篇代表作，现代 3D AI 辅助药物发现（AIDD）与先导化合物结构优化（Lead Optimization）已形成四项明确的物理建模共识。我们的 YLM 药物领域包（`src/toolkit/domains.py`）必须严格对齐这些输入约定、几何判据与评测标准：

### 1. 标准 3D 药效团分类学与方向几何公差（Pharmacophore Taxonomy & Cone Tolerances）
综合 **DEVELOP (#13)**、**PGMG (#24)**、**PharmacoNet (#26)** 与 **ShEPhERD (#30)**，行业通用的 3D 药效团并非无方向质点，而是**“标量类别（`0e`）+ 锚点坐标 + 方向张量（`1o`/`2e`）”**的复合体：
- **氢键供体（HBD，`1o` 极向量）**：锚定于重原子 $\text{D} \in \{\text{N, O, S}\}$，方向由共价键 $\hat{\mathbf{u}}_{\text{D}\to\text{H}}$ 定义；有效蛋白-配体氢键要求重原子距离 $d(\text{D},\text{A}) \in [2.5, 3.5]\text{ \AA}$ 且供体-氢-受体夹角 $\angle(\text{D–H}\cdots\text{A}) \ge 120^\circ$（强氢键 $\ge 150^\circ$，圆锥半角容差 $\theta_{\text{tol}} \approx 30^\circ\text{–}60^\circ$）。
- **氢键受体（HBA，`1o` 孤对电子轴向/角平分线向量）**：锚定于含孤对电子的杂原子（羰基 $\text{C=O}$、吡啶/亚胺 $\text{N}$、醚氧 $-\text{O}-$、磺酰基 $\text{S=O}$、腈基 $-\text{C}\equiv\text{N}$），方向沿重原子键角外侧孤对电子合向量 $-\sum_k \hat{\mathbf{u}}_{\text{A}\to\text{nb}_k} / \|\cdot\|$ 指向口袋供体。
- **芳香环 / $\pi$-体系（Aromatic Ring，`2e` 平面法向无迹对称张量 / 双侧 $\pm\hat{\mathbf{n}}$）**：锚定于 5/6 元共轭环质心 $\mathbf{c}_{\text{ring}}$。注意环平面法向量 $\hat{\mathbf{n}} = \frac{(\mathbf{r}_2-\mathbf{r}_1)\times(\mathbf{r}_3-\mathbf{r}_1)}{\|\cdot\|}$ 在未定向时具有 **$\pm\hat{\mathbf{n}}$ 符号任意性**（**ShEPhERD #30** 专门指出必须对环法向做 $\pm\mathbf{v}$ 对称不变或二阶张量处理！在 $O(3)$ 不可约表示中，$\mathbf{Q}_{\text{ring}} = \hat{\mathbf{n}}\otimes\hat{\mathbf{n}} - \frac{1}{3}\mathbf{I}$ 是精确无符号歧义的 **`2e` 偶张量**；若口袋侧存在明确 $\pi$-堆积芳环或阳离子 $\text{Lys}^+/\text{Arg}^+$，则可定向为指向该口袋互作中心的 `1o` 互作极向量）。
- **疏水中心（Hydrophobic，`0e` 球对称簇）**、**正/负可电离中心（Pos/Neg Ionizable，如季铵/脒基 vs. 羧酸/四氮唑，`0e` 电荷 + 局部静电场 `1o` 梯度）** 及 **卤键 $\sigma$-hole（Halogen Bond，$\text{C–X}$ 延长线方向 `1o` 极向量，$\text{X} \in \{\text{Cl, Br, I}\}$，夹角 $\approx 160^\circ\text{–}180^\circ$）**。

### 2. 出射向量（Exit Vector）立体净空度、杂化键角与子口袋互补判据
综合 **DeLinker (#12)**、**3DLinker (#15)**、**DecompDiff (#6)**、**LinkerNet (#18)**、**DiffLinker (#19)** 与 **PoseBusters** 物理检验规范：
- **单点碰撞检测不足以表征真实生长通道**：取代基（R-group）或连接子从母核重原子锚点 $\mathbf{r}_{\text{anchor}}$ 沿单位出射向量 $\hat{\mathbf{u}}_{\text{exit}}$ 延伸时，第一个新成键重原子位于 $s \approx 1.40\text{–}1.54\text{ \AA}$（$\text{C}_{sp^2}\text{–C}$ 或 $\text{C}_{sp^3}\text{–C}$），而支链主体占据 $s \in [1.5, 4.5]\text{ \AA}$ 的圆锥走廊。
- **范德华（vdW）碰撞下界与子口袋“甜区（Sweet-Spot）”**：若沿射线任意采样步 $s$ 处与口袋重原子距离小于范德华接触下限（通常重原子间 $d < 2.2\text{–}2.5\text{ \AA}$，或扣除 vdW 半径后重叠 $> 0.4\text{ \AA}$），属于**立体碰撞禁区（Steric Clash）**；反之，若在 $s \in [2.8, 4.8]\text{ \AA}$ 完全无任何口袋原子（距离 $> 8\text{ \AA}$），则指向**纯溶剂暴露区（Solvent-Exposed）**，虽无碰撞但难以贡献额外疏水/极性结合亲和力；最具优化价值的生长向量指向**未填满的次级子口袋腔隙（Sub-pocket Cavity，最近邻口袋距离约 $3.2\text{–}5.5\text{ \AA}$）**（对应 **DecompDiff #6** 的 Arm 先验）。
- **无显式氢输入兼容与双锚点相对几何**：真实晶体复合物（PDB / CrossDocked2020）常省略非极性氢（仅含重原子）。算法必须支持从重原子配位数与局部共面/四面体几何反推开放价态（open-valence）出射向量；当存在两个待桥接片段时，标准几何描述符为锚点间距 $d$、两侧出射向量与连心线夹角 $(\theta_1, \theta_2)$ 及二面角 $\phi$（**DeLinker #12 / LinkerNet #18** 标准四元组）。

### 3. 配体–蛋白口袋–辅因子/金属离子的显式角色解耦（Role Separation）
综合 **Pocket2Mol (#1)**、**TargetDiff (#4)**、**RFAA (#27)**、**Boltz-1/2 (#29)** 与 **PocketXMol (#11)**：
- 结合位点裁剪半径通常定义为配体重原子周围 **$6.0\text{–}10.0\text{ \AA}$** 的蛋白残基全原子集合；
- 必须严格区分原子角色（`role`）：`role=0`（配体小分子/片段母核）、`role=1`（催化/结构金属离子，如 $\text{Zn}^{2+}, \text{Mg}^{2+}$，具有强配位方向性）、`role=2`（蛋白口袋残基原子或结构水分子）。若将口袋与配体混为同一角色，C–H 出射点与药效团识别会把蛋白侧链苯丙氨酸/酪氨酸误判为配体可修饰位点。

### 4. 评测与物理合理性对齐指标
除了传统 Vina 对接打分、QED 与合成可及性（SA）外，2024–2026 年主流工作（**MolCRAFT #8, VoxBind #9, SurfDock #10, ShEPhERD #30**）全面转向**几何与物理真实性指标**：分子内/分子间范德华碰撞率（Steric Clash Ratio）、芳环平面度偏差、氢键供受体几何满足率、三维静电互补性与 **O(3) 旋转/反演协变一致性误差**。

---

## 三、 创新性与学术空白深度剖析（Creativity & Whitespace Analysis）

通过审视上述四大簇 40 篇代表性文献，我们可以清晰定位当前三维药物设计与分子大模型领域存在的一道**结构性鸿沟（Architectural Chasm）**——这恰恰构成了 YLM 将 **“自然语言语义查询 + `IrrepsToText` $O(3)$-等变 `1o`/`2e` 方向交叉注意力 + `<vec>` 三维出射/药效团向量直接读出”** 相结合的核心原创价值。

### 1. 纯 3D 等变扩散/生成模型（Cluster A, B, C）的盲区：有几何对称性，无语言推理与交互诊断能力
以 **Pocket2Mol (#1)**、**TargetDiff (#4)**、**DiffDock (#5)**、**DiffLinker (#19)**、**SurfDock (#10)** 与 **ShEPhERD (#30)** 为代表的几何深度学习模型，通过 GVP、EGNN 或 `e3nn` 球谐不可约表示（`0e + 1o + 2e`）严格保持了 $SE(3)/O(3)$ 等变性。然而：
1. **缺乏开放词汇自然语言控制与多约束推理**：药物化学家在实际先导化合物优化中面对的是高度语义化、多条件耦合的问题——例如：*“在保持铰链区（hinge）氢键供体锚点不动的前提下，配体的净极性偶极矩指向哪里？哪一个芳环 C–H 出射向量既能避开口袋主链碰撞，又能朝向疏水次级腔隙生长？”* 纯 3D 扩散模型只能执行固定输入输出格式的黑盒随机采样（生成成百上千个候选分子再靠人工后处理筛选），**无法听懂自然语言语义查询，也无法对指定原子/基团的局部三维物理场进行交互式归因问答**。
2. **任务头高度割裂**：对接（DiffDock）、从头生成（TargetDiff）、连接子桥接（DiffLinker）与药效团对齐（ShEPhERD）各自需要专门设计的噪声调度与输出头，无法通过自然语言指令在同一个预训练权重上零样本/少样本切换查询目标（如在偶极向量、受力梯度、氢键方向、最优生长出射向量之间自由切换）。

### 2. 现有 3D 分子大语言模型（Cluster D）的致命瓶颈：$O(3)$ 方向张量在跨模态桥接处彻底坍缩！
反观试图引入自然语言推理能力的 Cluster D（#31–#40），现有全部技术路线在处理三维方向性物理量时均陷入了**两大有损瓶颈之一**：

| 现有 3D Mol-LLM 范式 | 代表论文 | 跨模态融合机制 | 数学对称性与方向性缺陷 |
| :--- | :--- | :--- | :--- |
| **范式 I：标量投影器瓶颈（Q-Former / MLP Scalar Bottleneck）** | **3D-MoLM (#35)**, **MolCA (#33)**, **DrugChat (#32)**, **3D-MolT5 (#40)** | 先用 `Uni-Mol`（基于成对距离矩阵的 $O(3)$-**不变**编码器）或 `E3FP` 提取特征，再经 Q-Former / MLP 投影为 LLM 词嵌入空间中的纯标量（`0e`）软提示 token | 设分子/复合物整体经历正交旋转或反射 $X \mapsto X Q^\top\ (Q \in O(3))$。由于编码器与 Q-Former 输出满足严格**不变性** $f(X Q^\top) = f(X)$，输入给 LLM 的所有软提示在任意旋转下完全恒等！因此，模型最多只能回答*偶极矩模长是多少德拜*或*HOMO-LUMO 能隙是多少 eV* 等 `0e` 标量问题，**在信息论与群表示论意义上绝对不可能读出随分子同步旋转的 `1o` 三维极向量（偶极方向、出射向量、原子力）或 `2e` 环平面法向**！ |
| **范式 II：离散坐标词元化瓶颈（Quantized XYZ / String Tokenization）** | **Lingo3DMol (#38)**, **BindGPT (#39)**, **TamGen (#21)** | 将笛卡尔坐标 $(x,y,z)$ 按 $0.1\text{ \AA}$ 网格量化为文本 token（如 BindGPT），或在局部标架下拼接离散球坐标 token（如 Lingo3DMol） | 直接将坐标写成离散文本彻底破坏了连续李群 $O(3)$ 结构：对输入施加一个任意非 $90^\circ$ 旋转 $Q$，原本的词元序列会发生剧烈跳变，模型既存在网格量化台阶误差，也无法保证输出向量满足 $\hat{\mathbf{v}}(X Q^\top) = Q \hat{\mathbf{v}}(X)$（旋转协变）及宇称反演法则 $\hat{\mathbf{v}}(-X) = -\hat{\mathbf{v}}(X)$（区分 `1o` 极向量与 `1e` 轴向量）。 |

### 3. YLM 的独特定位：`IrrepsToText` + `<vec>` 填补“语义可控 $\times$ 精确 $O(3)$ 张量协变读出”空白
YLM 架构恰好处于 Cluster A/B/C（严格等变几何）与 Cluster D（自然语言推理）的正交交汇空白区：
1. **输入桥接保真（`IrrepsToText` $O(3)$-等变 `1o`/`2e` 交叉注意力）**：不把三维等变骨干的高阶球谐特征（`0e` 标量 + `1o` 极向量 + `2e` 四极/平面张量）强行池化压扁成纯 `0e` Q-Former 标量，而是在保持各阶不可约表征（`irreps`）方向通道的前提下与语言模型隐状态建立耦合：语言 token 负责承载自然语言语义条件（如定位“配体极性偶极”或“空间无碰撞取代基生长方向”），而几何流精确维护三维参考系。
2. **输出协变保证（`<vec>` 结构化向量槽位 + 等变解码头）**：当 LLM 在语义推理流中生成特殊词元 `<vec>` 时，系统并非用自回归文本硬拼 `(x, y, z)` 数字字符串，而是由锚定于语义上下文的 **$O(3)$-等变 `1o` 向量读出头**直接解码连续三维物理向量 $\hat{\mathbf{v}} \in \mathbb{R}^3$，在机器精度上严格满足：
   $$\forall Q \in O(3), \quad \mathbf{v}_{\text{pred}}(X Q^\top, \text{prompt}) = Q \, \mathbf{v}_{\text{pred}}(X, \text{prompt})$$
   这使 YLM 成为首个既能理解自然语言药物化学查询、又能精确输出旋转/反演协变三维出射向量与药效团极性向量的交互式探针（Probe）框架。

---

## 四、 基于文献调研的 YLM 药物模块（`src/toolkit/domains.py`）5 项具体设计升级建议

审视当前 `src/toolkit/domains.py` 中的 `analyze_drug_geometry()` 与 `enrich_domain_result()` 实现（第 256–372 行），现有逻辑仅遍历显式氢原子（`Z == 1`），仅识别 `N-H`/`O-H` 氢键供体，且仅在单一距离步长 `probe_step_ang = 2.4 Å` 处对显式 `C-H` 键做最近邻距离碰撞过滤。结合上述 40 篇文献的核心设计思想，建议实施以下 **5 项高价值、零外部重依赖（纯 NumPy 精确解析几何，严格保持 $O(3)$ 协变性）的具体升级**：

### 升级 1：引入芳香环检测与 `2e` 四极平面张量 / 口袋定向 $\pi$-法向特征（对齐 ShEPhERD #30, PharmacoNet #26, SurfDock #10）
- **现状痛点**：当前代码完全忽略了药物分子中最核心的骨架单元——5 元/6 元芳环与杂芳环（苯环、吡啶、嘧啶、吲哚、噻唑等），且若直接取叉积法向量 $\hat{\mathbf{n}} = \mathbf{e}_1 \times \mathbf{e}_2$，会随原子编号顺序翻转正负号（属于 `1e` 赝向量且具 $\pm\hat{\mathbf{n}}$ 歧义）。
- **具体升级方案**：
  1. 基于共价键邻接图检测配体内的 5/6 元无桥共面环（SVD 拟合环原子坐标中心化矩阵，最小奇异值对应平面法向 $\hat{\mathbf{n}}$，要求平面均方根偏差 $\text{RMSD}_{\text{plane}} < 0.12\text{ \AA}$）。
  2. 在 `domain_summary["aromatic_rings"]` 中同时输出：
     - **环质心锚点** $\mathbf{c}_{\text{ring}} \in \mathbb{R}^3$；
     - **消除 $\pm\hat{\mathbf{n}}$ 符号歧义的二阶无迹对称偶张量（`2e` representation）**：
       $$\mathbf{T}^{(2e)}_{\text{ring}} = \hat{\mathbf{n}}\otimes\hat{\mathbf{n}} - \frac{1}{3}\mathbf{I}_3$$
       该张量在任意 $Q \in O(3)$（含镜面反射！）下严格满足 $\mathbf{T}^{(2e)} \mapsto Q \mathbf{T}^{(2e)} Q^\top$ 且对 $\hat{\mathbf{n}} \leftrightarrow -\hat{\mathbf{n}}$ 完全不变；
     - **口袋定向 $\pi$-互作 `1o` 向量**：当存在蛋白口袋（`pocket_idx` 非空）时，将 $\pm\hat{\mathbf{n}}$ 定向为朝向 $5.5\text{ \AA}$ 内最近口袋芳环中心或阳离子氮原子（$\text{Lys/Arg}$，阳离子-$\pi$ 作用）的一侧，作为精确 `1o` 极向量输出。

### 升级 2：补全氢键受体（HBA 孤对电子轴向向量）与供体–受体圆锥角（Cone-Angle）互补评分（对齐 PharmacoNet #26, DEVELOP #13, RFAA #27）
- **现状痛点**：`analyze_drug_geometry()` 目前只统计 `hbond_donors`（`N-H`/`O-H`），遗漏了药物中数量更多且无需连接氢原子的**氢键受体（HBA）**（如酰胺/酮羰基氧 `=O`、吡啶环氮 `:N`、磺酰氧 `S=O`、醚氧 `-O-`、腈氮 `≡N`），也未检验配体与口袋之间是否形成真实氢键锚定。
- **具体升级方案**：
  1. 对配体中配位数 $1 \le k \le 3$ 的电负性重原子 $\text{A} \in \{\text{N, O, F, S}\}$（排除季铵正氮与吡咯型三配位共面内氢氮），计算其所有成键重/氢邻居单位向量的反向角平分线：
     $$\hat{\mathbf{u}}_{\text{HBA}}(i) = -\frac{\sum_{j \in \mathcal{N}(i)} (\mathbf{r}_j - \mathbf{r}_i)/\|\mathbf{r}_j - \mathbf{r}_i\|}{\left\|\sum_{j \in \mathcal{N}(i)} (\mathbf{r}_j - \mathbf{r}_i)/\|\mathbf{r}_j - \mathbf{r}_i\|\right\|}$$
     该公式对端基羰基 `C=O`（$k=1$，沿 $\text{C}\to\text{O}$ 延长线）、吡啶氮 `:N`（$k=2$，沿环外孤对电子方向）和醚氧 `-O-`（$k=2$，角平分线外指）天然 $O(3)$-等变！
  2. 当存在口袋原子时，增加**口袋-配体氢键锚点检测（`pocket_hbonds`）**：要求供体-受体距离 $d_{\text{DA}} \in [2.4, 3.5]\text{ \AA}$ 且方向圆锥余弦 $\cos\theta = \hat{\mathbf{u}}_{\text{donor/HBA}} \cdot \hat{\mathbf{r}}_{\text{lig}\to\text{pocket}} \ge 0.5$（即圆锥半角 $\le 60^\circ$），并在先导化合物优化时将已形成关键氢键的 `N-H`/`O-H` 标记为 `pharmacophore_locked = True`，防止误推荐破坏关键铰链区氢键的修饰位点。

### 升级 3：无氢重原子开放价态推断 + 多步范德华射线行进（Ray-Marching）与环平面/键轴夹角过滤（对齐 DeLinker #12, DecompDiff #6, DiffLinker #19, SurfDock #10）
- **现状痛点**：① 若输入为缺少显式氢的 PDB/SDF 重原子结构，`zi == 1` 循环直接返回空的 `exit_vectors`；② 单点 `2.4 Å` 探测可能跨过 `1.5 Å` 近程碰撞墙或落入封闭微孔而忽略 `3.6 Å` 处的主链阻挡。
- **具体升级方案**：
  1. **重原子自适应出射向量生成**：当配体碳/氮原子连接显式氢时保留精确 $\text{X}\to\text{H}$ 向量；当无显式氢输入时，对配位数为 2 的芳环/烯碳（$sp^2$）沿外角平分线 $-\text{norm}(\hat{\mathbf{b}}_1 + \hat{\mathbf{b}}_2)$ 构造平面内开放出射向量，对端基甲基/亚甲基按局部标架补齐开放价方向。
  2. **多步 vdW 射线行进净空剖面（Multi-step Ray-Marching Clearance Profile）**：沿候选出射方向 $\mathbf{p}(s) = \mathbf{r}_{\text{anchor}} + s\,\hat{\mathbf{u}}_{\text{exit}}$ 在采样序列 $s \in \{1.5, 2.2, 3.0, 3.8, 4.5\}\text{ \AA}$ 上计算扣除原子范德华半径后的有效净空度：
     $$c_{\text{vdW}}(s) = \min_{j \in \mathcal{P} \cup (\mathcal{L}\setminus\mathcal{N}_1)} \Big(\|\mathbf{r}_j - \mathbf{p}(s)\| - r_{\text{vdW}}(Z_j)\Big)$$
     过滤掉近程碰撞（$\min_{s \le 3.0} d(s) < 2.2\text{ \AA}$），并区分**“子口袋结合通道（Sub-pocket Channel，远端口袋距离 $3.2\text{–}5.5\text{ \AA}$）”**与**“溶剂暴露通道（Solvent-Exposed，远端距离 $> 6.0\text{ \AA}$）”**，综合净空度与子口袋接触潜力进行复合打分排序。
  3. **键轴/芳环平面夹角诊断**：记录每个出射向量与其母环平面或主干键轴的夹角余弦（如芳环面内取代 $\cos\alpha_{\text{plane}} \approx 1.0$ vs. 脂环轴向/平伏向取代），直接对齐 **DEVELOP (#13)** 与 **DecompDiff (#6)** 的几何特征。

### 升级 4：局部静电/极性互补 `1o` 向量（配体极性偶极估算与口袋局部静电场梯度对齐，对齐 ESP-Sim #22, ShEPhERD #30, IPDiff #25）
- **现状痛点**：`analyze_drug()` 的默认提问包含 *“What is the polar dipole direction of the ligand and best substituent growth direction?”*，但 `analyze_drug_geometry()` 与 `enrich_domain_result()` 目前仅附加了 `best_exit_vector`，缺乏一个物理可验证的配体极性偶极向量与口袋静电互补向量。
- **具体升级方案**：
  - 基于 Pauling 电负性差或轻量级拓扑部分电荷近似 $q_i$（质心中心化后消除净电荷原点依赖：$\boldsymbol{\mu}_{\text{lig}} = \sum_{i \in \mathcal{L}} (q_i - \bar{q})(\mathbf{r}_i - \bar{\mathbf{r}}_{\mathcal{L}})$）计算 $O(3)$-协变的配体极性方向 `ligand_polar_dipole_dir`（`1o` 极向量）；
  - 在最优出射点 $\mathbf{p}(2.4\text{ \AA})$ 处计算周围口袋极性重原子/带电残基产生的局部静电吸引场方向 $\mathbf{E}_{\text{pocket}}(\mathbf{p})$，据此给出取代基化学性质建议（若 $\mathbf{E}_{\text{pocket}}$ 强极性则提示引入极性 HBD/HBA 基团，若周围全为非极性碳硫原子则提示引入疏水/卤素基团）。

### 升级 5：双片段连接子（Linker / PROTAC）桥接几何诊断模式与子口袋角色细化（对齐 DeLinker #12, LinkerNet #18, DiffLinker #19）
- **现状痛点**：当输入 `StructureFrame` 中配体包含两个不连通的先导片段（FBDD fragment linking 或 PROTAC 弹头桥接场景）时，当前函数仅各自独立列出单侧出射向量，未计算跨片段桥接可行性。
- **具体升级方案**：
  - 利用配体共价距离矩阵自动检测连通分量数 $N_{\text{frag}}$；当 $N_{\text{frag}} \ge 2$ 时，自动配对不同片段上的无碰撞出射向量 $(\hat{\mathbf{u}}_a, \hat{\mathbf{u}}_b)$，输出 **DeLinker / LinkerNet 标准桥接四元组** $(d_{ab}, \cos\theta_a, \cos\theta_b, \phi_{\text{dihedral}})$ 以及连接直线走廊的中点口袋净空度 `bridge_corridor_clearance_angstrom`，并在 `enrich_domain_result()` 中额外挂载指向对侧锚点的 `linker_bridge_vector`（`1o`），无缝拓展 YLM 药物工具包对片段连接与 PROTAC 设计的支持！

---

## 五、 总结对照：YLM 药物模块升级前后能力矩阵

| 维度 | 传统 3D 扩散生成模型 (Cluster A–C) | 现有 3D 分子 LLM (Cluster D, 如 3D-MoLM) | YLM `domains.py` 现行基线 | 融合文献升级后的 YLM 药物模块 |
| :--- | :---: | :---: | :---: | :---: |
| **开放式自然语言 SAR/优化查询** | 不支持 | 支持 | **支持** | **支持（含极性/氢键/子口袋语义）** |
| **精确连续 $O(3)$ 向量协变输出 (`<vec>`)** | 仅隐式采样坐标 | **不支持（Q-Former `0e` 坍缩或网格量化失真）** | **支持（`1o`）** | **支持（`1o` 极向量 + `2e` 芳环张量诊断）** |
| **氢键供体（HBD）+ 受体（HBA）圆锥几何** | 部分支持（如 ShEPhERD） | 不支持 | 仅显式 HBD | **全支持（HBD + 孤对电子 HBA + 口袋圆锥角）** |
| **无显式氢 PDB/SDF 出射向量鲁棒提取** | 支持 | 不支持 | 不支持（依赖 `Z==1`） | **支持（重原子杂化开放价态自动反推）** |
| **多步 vdW 射线行进与子口袋甜区识别** | 隐式学习 | 不支持 | 仅 `2.4 Å` 单点 | **支持（`1.5–4.5 Å` 剖面 + 子口袋 vs. 溶剂区分）** |
| **双片段 Linker / PROTAC 相对出射角诊断** | 专用模型支持（DiffLinker/LinkerNet） | 不支持 | 不支持 | **原生支持（连通分量自动触发桥接四元组）** |
