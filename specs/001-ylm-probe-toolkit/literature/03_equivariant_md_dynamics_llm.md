# 等变分子动力学（Equivariant MD）、速度/位移几何图网络、反应过渡态/质子传递轨迹分析与分子动力学大模型智能体文献综述（2020–2026）

> **调研范围与验证说明**：本报告严格遴选并核验了 **2020–2026 年间发表于顶级同行评审期刊/会议（Nature, Nature Chemistry, Nature Communications, Nature Computational Science, Nature Machine Intelligence, PNAS, Chemical Science, JCTC, JCP, ICML, NeurIPS, ICLR, CVPR, AAAI, EMNLP, ACL）及 arXiv 预印本的 40 篇真实代表性文献**，覆盖四大技术子簇（每簇各 10 篇，合计 40 篇）：
> - **Cluster A（No. 1–10）**：等变神经力场与张量原子间势（Equivariant Neural Force Fields & Tensorial Potentials）
> - **Cluster B（No. 11–20）**：等变轨迹预测、显式速度/位移几何图网络与生成式分子动力学（Velocity-Equipped Geometric GNNs & Generative MD）
> - **Cluster C（No. 21–30）**：反应分子动力学、过渡态生成、Grotthuss 质子传递与稀有事件集体变量检测（Reactive MD, Transition-State / Proton-Transfer & Event Detection）
> - **Cluster D（No. 31–40）**：面向三维分子结构、力场与分子动力学模拟分析的大语言模型与多智能体（LLMs, Multimodal 3D-LLMs & Agents for MD）

---

## 0. 核心洞察：分子动力学智能的“几何-动力学-语义”三重断层

当前分子模拟与人工智能交叉领域存在一个极其鲜明的结构性断层：
1. **“几何精确但语义失语”的等变计算底座（Clusters A–C）**：以 `NequIP`、`MACE`、`Allegro`、`ViSNet` 为代表的 $E(3)/SO(3)$ 等变机器学习原子间势（MLIPs）通过球谐函数 $Y_m^{(l)}$ 与 Clebsch-Gordan（CG）张量积捕获了高精度多体方向性相互作用，但在读出层将所有高阶不可约表示（irreps: `1o, 2e, 3o`）坍缩为单一标量势能 $E = \sum_i E_i$ 并通过自动微分求保守力 $\mathbf{F}_i = -\nabla_{\mathbf{r}_i} E$，**既丢弃了帧间位移/速度奇宇称矢量通道（`1o` $\Delta\mathbf{r}_i$），也完全不具备自然语言解释能力**；而以 `SEGNN`、`EGNN`、`Timewarp`、`MDGen`、`EquiJump` 为代表的动态/生成式轨迹模型虽然引入了速度 `1o` 或跨时间步转移算子，但输出仍停留在沉默的三维数值坐标序列。
2. **“具备语言能力但动力学失明”的外挂脚本型大模型（Cluster D）**：以 `ChemCrow`、`Coscientist`、`AtomAgents`、`MDCrow`、`MDAgent2` 为代表的 LLM 智能体通过编写或调用外部 Python/MDTraj/LAMMPS 脚本读取标量日志（如全局 RMSD、温度、预设氢键计数），大模型本身完全无法感知底层三维轨迹张量场；以 `3D-MoLM` 为代表的三维分子多模态大模型仅基于 $SE(3)$ **不变距离矩阵（`0e`）**编码**单一静态平衡构象**；即便是最新将 LLM 与等变性结合的 `EquiLLM`（ICML 2025），其架构方向也是 `3D -> 不变 LLM 特征处理器 -> 3D 坐标预测`，即将 LLM 禁锢在标量隐空间内用于预测未来三维坐标，而非将动态 `O(3)` 不可约表示与反应动力学功耦合直接投影为自然语言机理诊断。

因此，**围绕 `(pos, delta_pos_1o, force_1o)` 构筑事件引导（Event-Guided）的 `IrrepsToText` 跨模态桥梁，并在其中内嵌群论自洽的逐原子方向性机械功分解（$W_i = \mathbf{F}_i \cdot \Delta\mathbf{r}_i$）与反应集体变量（如 Grotthuss 质子共享坐标 $\delta = d_1 - d_2$）**，精准命中了 2020–2026 现有文献体系的交叉空白区。

---

## 1. 四大技术子簇演进脉络深度剖析

### 1.1 Cluster A：等变神经力场与张量原子间势（No. 1–10）
早期三维等变网络如 Tensor Field Networks（TFN, Thomas et al., 2018）与 Cormorant（Anderson et al., NeurIPS 2019）奠定了基于球谐基与 CG 张量积的 $SO(3)$ 表示框架。2020 年后，该方向实现了从“小分子验证”向“材料/生物大体系第一性原理动力学底座”的跨越：
- **标量-矢量双通道轻量化路线（$l \le 1$）**：`PaiNN`（Schütt et al., ICML 2021）与 `TorchMD-NET`（Thölke & De Fabritiis, ICLR 2022）在笛卡尔空间并行维护标量（`0e`）与三维矢量（`1o`）特征，通过矢量内积反馈至标量门控，兼顾偶极矩/极化率等张量性质预测与高推理帧率；`ViSNet`（Wang et al., Nat. Commun. 2024；并进一步支撑 `AI2BMD`, Wang et al., *Nature* 635, 1019–1027, 2024 的全原子蛋白质从头算动力学）引入运行时几何计算二面角与矢量-标量交互消息传递，在不显式构造高阶 CG 积的前提下捕获四体扭转效应。
- **高阶球谐张量积与多体完备展开路线（$l \ge 2$）**：`SpookyNet`（Unke et al., Nat. Commun. 2021）融合电子电荷/自旋态（$Q, S$）与非局域自注意力以刻画质子化/自由基体系；`NequIP`（Batzner et al., Nat. Commun. 2022）证明完整保留奇偶宇称的高阶 $E(3)$ 不可约表示（`0e, 1o, 1e, 2e, 3o`）可显著提升力场样本效率与非平衡构象泛化性；`MACE`（Batatia et al., NeurIPS 2022；及其通用底座 `MACE-MP-0`, Batatia et al., *J. Chem. Phys.* 163, 184110, 2025）结合原子簇展开（ACE），在单层内通过对称化高阶张量积直达四体及以上多体阶数，将消息传递压缩至 2 层；`Allegro`（Musaelian et al., Nat. Commun. 2023）剔除图消息传递，仅在严格局域有序原子对上执行高阶 $O(3)$ 张量积，支撑上亿原子规模 LAMMPS 并行模拟。
- **等变注意力与 $SO(2)$ 降维加速路线**：`SO3krates`（Frank et al., NeurIPS 2022）将不变欧氏特征与球谐坐标解耦以捕获长程相互作用；`Equiformer`（Liao & Smidt, ICLR 2023；后续扩展为 `EquiformerV2`, ICLR 2024）以深度张量积 MLP 注意力替代点积注意力；`eSCN`（Passaro & Zitnick, ICML 2023）利用键轴对齐旋转将 $O(L^6)$ 的 $SO(3)$ 卷积精确化简为 $O(L^3)$ 的 $SO(2)$ 卷积，使 $L_{\max}=6$ 以上的角向分辨率成为可能。
- **共性局限**：Cluster A 所有模型均为**单帧静态 Born-Oppenheimer 势能面求值器** $\mathbf{r}(t) \mapsto (E, -\nabla_{\mathbf{r}}E)$。它们既不接受帧间位移/速度 $\Delta\mathbf{r}_i$（`1o`），也不区分“热涨落高受力”与“有效跨垒做功”，更不具备任何自然语言交互或事件归因接口。

### 1.2 Cluster B：等变轨迹预测、速度几何图网络与生成式 MD（No. 11–20）
为突破飞秒积分步长限制或直接建模动力学演化，Cluster B 形成了两条截然不同的技术支流：
- **支流 B1：显式注入速度/位移 `1o` 矢量的几何动力学图网络**。`EGNN`（Satorras et al., ICML 2021）在其速度动力学公式中显式维护节点速度矢量 $\mathbf{v}_i \in \mathbb{R}^3$，利用相对位移加权和更新速度并推进位置；`SEGNN`（Brandstetter et al., ICLR 2022）从群表示论高度提出**可控 $E(3)$ 等变图网络**，将节点速度 $\mathbf{v}_i$、动量、力/加速度及边相对位移统一编码为奇宇称球谐张量（`1o` 及其高阶张量积分量），驱动 Steerable MLP 消息传递；`GMN`（Huang et al., ICLR 2022）在广义坐标下联合演化 $(\mathbf{x}, \mathbf{v})$ 以严格满足刚性键/铰链几何约束（但刚性拓扑假设使其无法处理断键/成键反应）；`ProtMD`（Wu et al., Adv. Sci. 2022）与 `DiffMD`（Wu & Li, AAAI 2023）则利用蛋白质构象序列及帧间速度/加速度条件引导的等变几何扩散模型预测后续构象帧。
- **支流 B2：跨时间尺度生成式转移算子与全轨迹流匹配**。`Timewarp`（Klein et al., NeurIPS 2023）通过条件归一化流直接采样跨越 $10^5\text{--}10^6\text{ fs}$ 的粗粒化时间跃迁构象；`ITO`（Schreiner et al., NeurIPS 2023）在 $SE(3)$ 等变扩散框架中引入时间滞后标量 $N\Delta t$ 学习多时间分辨率隐式转移算子；`GeoTDM`（Han et al., NeurIPS 2024）在整条三维时空轨迹分布上交替执行 $SE(3)$ 空间卷积与时间注意力；`MDGen`（Jing et al., 2024）将蛋白质全轨迹视作三维结构“视频”，通过 $SE(3)$ 不变残基内部坐标分词与流匹配统一支持前向模拟、过渡路径采样（TPS）与轨迹插值；`EquiJump`（Costa et al., 2024）则利用双端 $SO(3)$ 等变随机插值器（Stochastic Interpolants）直接跨越长间隔模拟步。
- **共性局限**：支流 B2 的大步长跳跃模型（`Timewarp`/`ITO`/`EquiJump`）直接抹平了飞秒级过渡态鞍点穿越细节与瞬时力-位移做功过程；支流 B1 的速度等变网络虽保留了 `1o` 运动学通道，但输出仍是沉默的未来坐标 $\hat{\mathbf{r}}(t+\Delta t)$，缺乏化学反应集体变量感知与文本化机理输出。

### 1.3 Cluster C：反应分子动力学、过渡态/质子传递与稀有事件检测（No. 21–30）
化学反应与质子传递属于典型的高能垒稀有事件，要求模型同时直面**非平衡电子结构重构**与**低维反应坐标（CV）提取**：
- **非平衡数据集与生成式过渡态求解**：`Transition1x`（Schreiner et al., Sci. Data 2022）构建了覆盖 10,073 个有机反应 NEB 最小能量路径的 960 万个非平衡 DFT 构象与力数据，打破了近平衡数据集（QM9/MD17）无法泛化至断键能垒区的瓶颈；`OA-ReactDiff`（Duan et al., Nat. Comput. Sci. 2023）提出对象感知 $SE(3)$ 等变扩散模型，在秒级内由反应物/产物生成高精度三维过渡态（TS）；`React-OT`（Duan et al., Nat. Mach. Intell. 2025）进一步以等变最优传输（Optimal Transport）替代随机扩散，沿确定性流路径实现更高精度与推理速度的 TS 生成；`EquiReact`/`3DReact`（van Gerwen et al., 2023 / JCIM 2024）则利用原子映射 $O(3)$ 等变张量网络直接预测活化能垒 $\Delta E^\ddagger$。然而，上述四者均面向**静态端点或静态鞍点搜索**，无法实时解析热涨落分子动力学轨迹中的动态过垒过程。
- **机器学习慢模态与稀有事件路径采样**：`Deep-TICA`（Bonati, Piccini & Parrinello, PNAS 2021）与统一开源框架 `mlcolvar`（Bonati et al., JCP 2023）基于时间滞后转移算子特征函数或判别式边界学习可微慢集体变量 $s(\mathbf{r})$ 及光滑配位数；Jung, Covino, Bolhuis & Hummer（Nat. Comput. Sci. 2023）将深度学习预测的提交者概率（Committor $p_B \in [0,1]$，其中 $p_B \approx 0.5$ 精确对应过渡态系综）与过渡路径采样（TPS）闭环耦合，自动发现分子自组装与反应机理。但这类模型将高维三维几何压缩为黑盒标量 `0e` 隐变量，丢失了逐原子力-位移方向信息，仍需人工事后做符号回归才能理解物理含义。
- **Grotthuss 质子传递与水相反应动力学的物理坐标范式**：Calegari Andrade, Car & Selloni（Chem. Sci. 2020）利用深度势能分子动力学解析水–$\text{TiO}_2$ 界面质子转移自由能，确立了**质子转移不对称伸缩坐标 $\delta = d(\text{O}_1\text{--H}) - d(\text{O}_2\text{--H})$** 与光滑氧-氢配位数 $s_{\text{OH}}$ 在机器学习反应 MD 中的标准地位；`ZundEig`（Di Pino et al. & Hassanali, JPCB 2023）对酸性水溶液轨迹开展无监督聚类，揭示了过剩质子在定域 Eigen 态（$\text{H}_9\text{O}_4^+, |\delta| > 0.3\text{ \AA}$）与对称共享 Zundel 过渡态（$\text{H}_5\text{O}_2^+, |\delta| \approx 0$）之间的连续涨落谱；Gomez, Thompson & Laage（*Nature Chemistry* 2024）结合全原子核量子效应神经网络 MD，证明液态水中的 Grotthuss 质子跳跃受控于供体端与受体端氢键顺序交换的“双重门控（doubly gated）”机制，有效区分了无净迁移的往返振荡（rattling）与真正的不可逆质子跃迁。

### 1.4 Cluster D：三维分子多模态模型与分子动力学智能体（No. 31–40）
大语言模型向分子科学渗透呈现出三条截然不同的架构路径，每一条都暴露出清晰的边界：
- **路径 D1：静态分子图/单构象跨模态投影（Graph/3D-to-Text）**。`MolCA`（Liu et al., EMNLP 2023）通过 Q-Former 将二维分子图投影至语言空间；`3D-MoLM`（Li et al., ICLR 2024）更进一步引入 Uni-Mol 编码三维分子构象并经 3D-Q-Former 对齐至 Llama-2。但 `3D-MoLM` 仅编码**单一静态平衡构象**，且其底层 Uni-Mol 完全依赖 $SE(3)$ **不变原子对距离矩阵（`0e`）**，既无 `1o`/`2e` 高阶等变方向表示，也完全无法处理多帧位移 $\Delta\mathbf{r}_i$、原子受力 $\mathbf{F}_i$ 与反应过渡态演化。
- **路径 D2：工具/脚本调用型化学与 MD 多智能体（Script-Calling Agents）**。`Coscientist`（Boiko et al., Nature 2023）与 `ChemCrow`（Bran et al., Nat. Mach. Intell. 2024）开创了基于文本工具调用的自主有机合成与化学规划；`ChatMol Copilot`（Sun et al., ACL 2024）通过命令行操纵 PyMOL 与分子对接脚本；面向原子级模拟的 `AtomAgents`（Ghafarollahi & Buehler, 2024 / PNAS 2025）、`MDCrow`（Campbell et al. & White, 2025）、量子化学智能体 `El Agente`（Zou et al. & Aspuru-Guzik, Matter 2025）以及专精模拟代码生成与领域问答的 `MDAgent2`（Shi et al., 2026；及同期的 `MDAgent`, Ma et al., 2026）均采用**“LLM 编写/触发 LAMMPS/OpenMM/MDTraj/ORCA 脚本 $\rightarrow$ 解析文本日志或静态二维统计图”**的间接交互范式。一旦轨迹中发生未预先硬编码进后处理脚本的瞬态局部重排、能垒攀爬恢复力异常或协同张量畸变，LLM 在三维几何与动力学层面上实际上是完全“失明”的。
- **路径 D3：内嵌 LLM 的等变几何预测器（`EquiLLM`, ICML 2025）**。Li, Cen, Huang et al.（ICML 2025）提出的 `EquiLLM`（*Large Language-Geometry Model: When LLM meets Equivariance*）首次尝试桥接 LLM 与 $E(3)$ 等变性，但其核心机制是将三维输入拆分为“几何感知标量提示（PCA/距离统计）”送入 LLM 做**不变特征处理**，同时由外部等变编码器与等变适配器单独维护方向矢量，最终**输出预测的三维坐标**（用于 MD17 轨迹预测、人体运动与抗体环区生成）。换言之，`EquiLLM` 把 LLM 当作 `3D -> 3D` 坐标回归器内部的标量算子，**既没有赋予模型面向人类科学家的自然语言反应机理输出能力（`IrrepsToText`），也将 LLM 与 `1o` 力-位移做功耦合彻底隔离**。

---

## 2. 40 篇真实核验核心文献全景对比总表（2020–2026）

| No. | 论文题目、作者与年份 / 发表载体（arXiv ID） | 输入几何与动力学特征（静态 $\mathbf{r}$ vs 速度/位移 $\Delta\mathbf{r}$ `1o`） | 输出类型（Output Type） | 语言交互与语义能力（Language Capability） | 核心局限性（Key Limitation） |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **Cluster A** | **等变神经力场与张量原子间势（10 篇）** | | | | |
| 1 | **PaiNN**: *Equivariant message passing for the prediction of tensorial properties and molecular spectra* — Schütt, Unke & Gastegger, **ICML 2021** (`arXiv:2102.03150`) | 静态构象 $(Z_i, \mathbf{r}_i)$；维护标量 `0e` 与笛卡尔单位键矢 `1o` ($\hat{\mathbf{r}}_{ij}$)，**无帧间 $\Delta\mathbf{r}$ `1o`** | 标量势能 $E$、保守力 $\mathbf{F}_i$、偶极矩 `1o`、极化率张量 `0e+2e` | 无（纯数值张量输出） | 截断于 $l=1$ 矢量通道，无时间演化记忆，无法输出动力学语义解释 |
| 2 | **SpookyNet**: *Learning force fields with electronic degrees of freedom and nonlocal effects* — Unke et al., **Nat. Commun. 2021**, 12, 7273 (`arXiv:2105.00304`) | 静态 $(Z_i, \mathbf{r}_i)$ + 总电荷 $Q$/自旋 $S$ (`0e`) + 局域球谐特征 ($l \le 2$)，**无 $\Delta\mathbf{r}$ `1o`** | 标量势能 $E$、保守力 $\mathbf{F}_i$、原子部分电荷与偶极矩 | 无 | 单帧静态势能面求值器；虽支持质子化电荷态，但无跨帧动力学与文本解读能力 |
| 3 | **NequIP**: *E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials* — Batzner et al., **Nat. Commun. 2022**, 13, 2453 (`arXiv:2101.03164`) | 静态 $(Z_i, \mathbf{r}_i)$；相对键矢投影至完整 $E(3)$ 球谐不可约表示 $Y_m^{(l)}(\hat{\mathbf{r}}_{ij})$ ($l \le 3$, 含 `e/o` 宇称)，**无 $\Delta\mathbf{r}$ `1o`** | 逐原子标量能量 $E_i$ (`0e`) 求和与自动微分力 $\mathbf{F}_i = -\nabla_{\mathbf{r}_i}E$ (`1o`) | 无（典型“沉默型”MLIP） | 读出层将内部高阶 irreps 全部坍缩为单一能量标量，丢弃帧间做功 $W_i$ 且无语言接口 |
| 4 | **TorchMD-NET**: *Equivariant Transformers for Neural Network based Molecular Potentials* — Thölke & De Fabritiis, **ICLR 2022** (`arXiv:2202.02541`) | 静态 $(Z_i, \mathbf{r}_i)$；`0e` 标量与 `1o` 矢量等变点积自注意力，**无 $\Delta\mathbf{r}$ `1o`** | 标量势能 $E$ 与保守原子力 $\mathbf{F}_i$ | 无 | 仅限 $l \le 1$ 静态快照编码，缺乏多体高阶角向张量及动态过垒事件语义感知 |
| 5 | **MACE**: *Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields* — Batatia et al., **NeurIPS 2022** (`arXiv:2206.07697`) | 静态 $(Z_i, \mathbf{r}_i)$；基于原子簇展开（ACE）对称化高阶张量积（`0e, 1o, 2e, 3o`），**无 $\Delta\mathbf{r}$ `1o`** | 标量势能 $E$、保守力 $\mathbf{F}_i$ 及维里应力张量 $\boldsymbol{\sigma}$ | 无（沉默型高阶力场） | 高体阶张量仅服务于瞬时静态能量回归，无法耦合帧间位移 `1o` 或输出机理文本 |
| 6 | **SO3krates**: *Equivariant attention for interactions on arbitrary length-scales in molecular systems* — Frank, Unke & Müller, **NeurIPS 2022** (`arXiv:2205.14276`) | 静态 $(Z_i, \mathbf{r}_i)$；解耦欧氏不变特征与 `SO(3)` 球谐坐标分支并引入滤波注意力，**无 $\Delta\mathbf{r}$ `1o`** | 标量势能 $E$ 与保守力 $\mathbf{F}_i$ | 无 | 仅建模单帧空间长程关联，缺失时间反演奇性速度特征与自然语言问答能力 |
| 7 | **Allegro**: *Learning local equivariant representations for large-scale atomistic dynamics* — Musaelian et al., **Nat. Commun. 2023**, 14, 579 (`arXiv:2204.05249`) | 严格局域静态有序原子对几何 $\mathbf{r}_{ij}$ 的高阶 $O(3)$ 张量积（无消息传递），**无 $\Delta\mathbf{r}$ `1o`** | 局域原子对能量 $E_{ij}$ (`0e`) 与保守力 $\mathbf{F}_i$ | 无 | 为超大规模空间并行牺牲跨节点与跨帧动态上下文，对反应事件语义完全无感 |
| 8 | **Equiformer**: *Equivariant Graph Attention Transformer for 3D Atomistic Graphs* — Liao & Smidt, **ICLR 2023** (`arXiv:2206.11990`) | 静态 $(Z_i, \mathbf{r}_i)$；`SO(3)` irreps 结合深度张量积（DTP）MLP 注意力，**无 $\Delta\mathbf{r}$ `1o`** | 标量能量 $E$、原子力 $\mathbf{F}_i$ 及 QM9 量子化学标量属性 | 无 | 纯静态三维图注意力架构，未建模时间演化位移 `1o` 与力做功耦合，无文本生成 |
| 9 | **eSCN**: *Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs* — Passaro & Zitnick, **ICML 2023** (`arXiv:2302.03655`) | 静态 $(Z_i, \mathbf{r}_i)$；沿键轴旋转对齐将高阶 $SO(3)$ 张量卷积精确约化为 $SO(2)$ 卷积 ($L_{\max}\ge 6$)，**无 $\Delta\mathbf{r}$ `1o`** | 催化吸附体系/分子标量能量 $E$ 与原子力 $\mathbf{F}_i$ | 无 | 极致优化单帧高角动量卷积效率，但不具备多帧运动学表征与自然语言读出机制 |
| 10 | **ViSNet**: *Enhancing geometric representations for molecules with equivariant vector-scalar interactive message passing* — Wang et al., **Nat. Commun. 2024**, 15, 313 (`arXiv:2210.16518`) | 静态 $(Z_i, \mathbf{r}_i)$；运行时提取键角/二面角标量并与 `1o` 矢量交互更新，**无 $\Delta\mathbf{r}$ `1o`** | 标量能量 $E$、保守力 $\mathbf{F}_i$ 与分子电子性质 | 无 | 仍局限于单帧静态构象求值，驱动 MD 时需依赖外部人工脚本分析轨迹事件 |
| **Cluster B** | **等变轨迹预测、速度几何图网络与生成式 MD（10 篇）** | | | | |
| 11 | **EGNN**: *E(n) Equivariant Graph Neural Networks* — Satorras, Hoogeboom & Welling, **ICML 2021** (`arXiv:2102.09844`) | 节点标量 $\mathbf{h}_i$ (`0e`)、位置 $\mathbf{x}_i$ **及显式速度矢量 $\mathbf{v}_i \in \mathbb{R}^3$ (`1o`)** | 下一时刻三维坐标 $\mathbf{x}_i(t+\Delta t)$ 与更新速度 $\mathbf{v}_i(t+\Delta t)$ | 无 | 仅含标量距离与 $l=1$ 坐标线性组合，缺失 $l\ge 2$ 角向多体分辨力、功分解及语言能力 |
| 12 | **SEGNN**: *Geometric and Physical Quantities Improve E(3) Equivariant Message Passing* — Brandstetter, Hesselink, van der Pol, Bekkers & Welling, **ICLR 2022** (`arXiv:2110.02905`) | 位置 $\mathbf{x}_i$ + **节点速度 $\mathbf{v}_i$ (`1o`)、相对位移 $\mathbf{x}_i-\mathbf{x}_j$ (`1o`)、力/加速度 (`1o`)** 投影至球谐可控特征 ($l \le l_{\max}$) | 未来三维位移/位置或可控高阶物理张量场 | 无 | 确立了 `1o` 速度/位移注入 $E(3)$ 消息传递的黄金标准，但纯数值输出，无反应 CV 与语言投影 |
| 13 | **GMN**: *Equivariant Graph Mechanics Networks with Constraints* — Huang, Han, Rong, Xu, Sun & Huang, **ICLR 2022** (`arXiv:2203.06442`) | 位置 $\mathbf{x}_i$、**速度矢量 $\mathbf{v}_i$ (`1o`)** 及刚性几何约束（键杆/铰链广义坐标） | 满足几何约束的未来多体位置与速度轨迹 $(\mathbf{x}, \mathbf{v})$ | 无 | 预设固定刚性键拓扑约束，从根本上无法建模断键/成键反应（如质子转移），且无文本接口 |
| 14 | **ProtMD**: *Pre-Training of Equivariant Graph Matching Networks with Conformation Flexibility for Drug Binding* — Wu et al., **Adv. Sci. 2022**, 9(33), e2203796 | 蛋白质-配体 MD 构象帧序列 $(Z_i, \mathbf{r}_i^{(t)})$ 与**跨帧构象位移匹配信号 $\Delta\mathbf{r}_i$ (`1o`)** | 未来帧三维构象去噪还原及药物结合亲和力/动力学标量 | 无 | 将 MD 轨迹仅作为自监督预训练信号以服务下游结合能回归，不支持在线反应事件语言解读 |
| 15 | **DiffMD**: *A Geometric Diffusion Model for Molecular Dynamics Simulations* — Wu & Li, **AAAI 2023** (`arXiv:2204.08672`) | 历史三维构象帧 + **帧间速度/位移 $\Delta\mathbf{r}_i$ 与条件原子加速度矢量 (`1o`)** | 基于得分扩散采样的下一时间步分子构象 $\mathbf{r}(t+\Delta t)$ | 无 | 绕过势能面直接生成带噪坐标，丢失精确保守力做功 $W_i = \mathbf{F}_i \cdot \Delta\mathbf{r}_i$，无语言输出 |
| 16 | **Timewarp**: *Transferable Acceleration of Molecular Dynamics by Learning Time-Coarsened Dynamics* — Klein et al., **NeurIPS 2023** (`arXiv:2302.01170`) | 当前全原子三维构象 $\mathbf{r}(t)$（辅助速度高斯边缘化；条件归一化流建模粗粒化大位移 $\Delta\mathbf{r}$） | 跨越 $10^5\text{--}10^6\text{ fs}$ 的玻尔兹曼分布候选构象 $\mathbf{r}(t+\tau)$ | 无 | 大步长跃迁直接跳过飞秒级过渡态鞍点穿越全过程，抹除瞬时速度与做功路径，无文本解释 |
| 17 | **ITO**: *Implicit Transfer Operator Learning: Multiple Time-Resolution Models for Molecular Dynamics* — Schreiner, Winther & Olsson, **NeurIPS 2023** (`arXiv:2305.18046`) | 初始三维构象 $\mathbf{r}(t)$ + 多尺度滞后时间标量 $N\Delta t$（基于 PaiNN 的 $SE(3)$ 等变扩散） | 条件转移概率密度采样构象 $p(\mathbf{r}_{t+N\Delta t}\mid \mathbf{r}_t)$ | 无 | 马尔可夫状态间密度采样器，不包含瞬时速度-受力方向功率分析，无自然语言交互能力 |
| 18 | **GeoTDM**: *Geometric Trajectory Diffusion Models* — Han, Xu, Lou, Ye & Ermon, **NeurIPS 2024** (`arXiv:2410.13027`) | 连续多帧三维时空轨迹序列 $\mathbf{x}_{1:T}$（交替执行 $SE(3)$ 空间卷积与时序注意力，隐含帧间 `1o` 位移） | 联合生成的整段三维运动/分子动力学轨迹序列 | 无 | 专注于全轨迹无条件/条件几何分布生成，缺乏原子受力物理做功拆解与反应机理文本化输出 |
| 19 | **MDGen**: *Generative Modeling of Molecular Dynamics Trajectories* — Jing, Stärk, Jaakkola & Berger, **arXiv 2024** (`arXiv:2409.17808`) | 初始/两端三维构象经 $SE(3)$ 不变内部扭转角相对参数化编码的时空轨迹 Token 序列 | 前向轨迹生成、过渡路径采样（TPS）及轨迹时间超分辨插值 | 无 | 采用残基级固定共价拓扑与内部扭转角分词，无法表征共价断键/质子跳跃，且无自然语言读出 |
| 20 | **EquiJump**: *Protein Dynamics Simulation via SO(3)-Equivariant Stochastic Interpolants* — Costa et al., **arXiv 2024** (`arXiv:2410.09667`) | 全原子三维坐标与重原子相对于 $\text{C}_\alpha$ 的 `SO(3)` `1o` 局域相对矢量场 | 基于双端随机插值器的长步长全原子蛋白质动态构象序列 | 无 | 面向非反应性蛋白质构象采样，跳过亚皮秒反应微观做功细节，不具备文本诊断接口 |
| **Cluster C** | **反应 MD、过渡态/质子传递与稀有事件检测（10 篇）** | | | | |
| 21 | *Free energy of proton transfer at the water–TiO₂ interface from ab initio deep potential molecular dynamics* — Calegari Andrade, Ko, Zhang, Car & Selloni, **Chem. Sci. 2020**, 11, 2335–2341 | 静态三维快照 + **质子共享不对称伸缩坐标 $\delta = d_{\text{O}_1\text{H}} - d_{\text{O}_2\text{H}}$** 与光滑氧-氢配位数 $s_{\text{OH}}$ (`0e`) | 深度势能 $(E, \mathbf{F}_i)$ 及水-氧化物界面质子转移自由能曲线 $\Delta F(\delta)$ | 无 | 底层采用不变描述符（无高阶 `SO(3)` irreps 与 `1o` 速度耦合），完全依赖人工编写后处理脚本解读 |
| 22 | **Deep-TICA**: *Deep learning the slow modes for rare events sampling* — Bonati, Piccini & Parrinello, **PNAS 2021**, 118(44), e2113533118 (`arXiv:2107.03943`) | 时间滞后对 $(\mathbf{d}(t), \mathbf{d}(t+\tau))$ 的成对距离/配位数等 $SE(3)$ 不变标量描述符 (`0e`) | 转移算子慢模态特征函数对应的低维非线性集体变量 $\mathbf{s} \in \mathbb{R}^k$ (`0e`) | 无 | 将复杂的三维协同反应压缩为不可解释的黑盒标量 `0e` 隐变量，丢失方向性做功且无语义输出 |
| 23 | **Transition1x**: *Transition1x – a Dataset for Building Generalizable Reactive Machine Learning Potentials* — Schreiner, Bhowmik, Vegge, Busk & Winther, **Sci. Data 2022**, 9, 779 (`arXiv:2207.12858`) | 覆盖 10,073 个有机反应 NEB 路径的 960 万个远非平衡三维反应构象 $(Z_i, \mathbf{r}_i)$ | 跨越最小能量路径（MEP）与过渡态的 DFT 能量 $E$ 和原子力 $\mathbf{F}_i$ | 无 | 静态 NEB 几何路径基准（无有限温度热速度 $\mathbf{v}_i$ 与动力学往返重穿轨迹），无文本标注 |
| 24 | *Machine-guided path sampling to discover mechanisms of molecular self-organization* — Jung, Covino, Arjun, Leitold, Dellago, Bolhuis & Hummer, **Nat. Comput. Sci. 2023**, 3, 334–345 | 过渡路径采样（TPS）反应轨迹帧中的内部几何距离与配位环境特征 (`0e`) | 预测提交者概率（Committor $p_B(\mathbf{r}) \in [0,1]$，`0e`），精准定位 $p_B \approx 0.5$ 过渡态系综 | 无 | 模型仅输出单一标量概率 $p_B$，仍需人工事后做符号回归寻找物理坐标，无法直接用自然语言报告机理 |
| 25 | **mlcolvar**: *A unified framework for machine learning collective variables for enhanced sampling simulations: mlcolvar* — Bonati, Trizio, Rizzi & Parrinello, **J. Chem. Phys. 2023**, 159, 014801 (`arXiv:2305.19980`) | 成对距离、键角及可微光滑配位数 $C_i = \sum_j \frac{1-(r_{ij}/r_0)^n}{1-(r_{ij}/r_0)^m}$ (`0e`) | 接入 PLUMED 的可微标量集体变量 $s(\mathbf{r})$ 与偏置力 $-\nabla_{\mathbf{r}}V_{\text{bias}}(s)$ | 无 | 严格限制于标量不变特征管道，缺失 `1o` 速度/位移球谐张量耦合及自然语言事件归因能力 |
| 26 | **OA-ReactDiff**: *Accurate transition state generation with an object-aware equivariant elementary reaction diffusion model* — Duan, Du, Jia & Kulik, **Nat. Comput. Sci. 2023**, 3, 1033–1045 (`arXiv:2304.06174`) | 反应物与产物分子片段的三维坐标 $(Z_i, \mathbf{r}_i^R, \mathbf{r}_i^P)$（对象感知 $SE(3)$ 等变扩散） | 生成的三维过渡态鞍点几何结构 $\mathbf{r}_i^{\text{TS}}$ | 无 | 静态“端点到鞍点”生成器，不处理连续时间 MD 轨迹 $(\mathbf{r}(t), \Delta\mathbf{r}(t))$ 与力做功，无语言能力 |
| 27 | **ZundEig**: *The Structure of the Proton in Liquid Water From Unsupervised Learning* — Di Pino et al. & Hassanali, **J. Phys. Chem. B 2023**, 127(45), 9822–9832 | 酸性水 AIMD/MLMD 轨迹的局域 SOAP 描述符与**质子共享坐标 $\delta = d_{\text{O}_1\text{H}} - d_{\text{O}_2\text{H}}$** | Zundel ($|\delta|\approx 0$) 与 Eigen ($|\delta|>0.3\text{ \AA}$) 连续溶剂化态的无监督密度分簇 | 无 | 离线静态特征密度聚类，未引入 `1o` 速度-受力功率分析，不支持自然语言轨迹检索与问答 |
| 28 | **EquiReact / 3DReact**: *EquiReact: An equivariant neural network for chemical reactions* — van Gerwen et al. & Corminboeuf, **arXiv 2023** (`arXiv:2312.09298`; **JCIM 2024**, 64, 5771–5785) | 反应物与产物三维构象的 $O(3)$ 等变高阶张量原子映射消息传递 | 化学反应活化能垒 $\Delta E^\ddagger$ 与反应焓 $\Delta H_r$ (`0e`) | 无 | 仅从静态始末态回归标量能垒，不包含时间分辨轨迹演化、`1o` 位移矢量或文本机理描述 |
| 29 | *Neural-network-based molecular dynamics simulations reveal that proton transport in water is doubly gated by sequential hydrogen-bond exchange* — Gomez, Thompson & Laage, **Nat. Chem. 2024**, 16, 1838–1844 | 含核量子效应的神经网络水轨迹；追踪 **$\delta = r_{\text{O}_d\text{H}^*} - r_{\text{O}_a\text{H}^*}$**、$R_{\text{OO}}$ 与供/受体氢键配位数跳变 | 揭示液态水 Grotthuss 质子有效跳跃的“供体+受体双重氢键交换门控”物理机制与速率 | 无 | 物理判据极为深刻，但完全依赖专家定制的离线统计脚本，尚未沉淀为端到端等变-语言神经网络模块 |
| 30 | **React-OT**: *Optimal transport for generating transition states in chemical reactions* — Duan et al. & Kulik, **Nat. Mach. Intell. 2025**, 7, 615–626 (`arXiv:2404.13430`) | 反应物-产物线性插值初始猜想构象与 $SE(3)$ 等变最优传输流场（隐含几何位移流 `1o`） | 确定性快速生成的三维过渡态几何 $\mathbf{r}_i^{\text{TS}}$ | 无 | 聚焦气相/基元反应静态鞍点几何求解，无法在线诊断有限温度凝聚相 MD 轨迹中的恢复力/驱动功与语言问答 |
| **Cluster D** | **面向三维分子、力场与模拟分析的 LLM 及多智能体（10 篇）** | | | | |
| 31 | **MolCA**: *Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter* — Liu et al., **EMNLP 2023** (`arXiv:2310.12798`) | 静态二维分子拓扑图（GIN 编码）+ 文本提示（经 Q-Former 投影至 Galactica） | 自然语言分子结构描述、IUPAC 命名与性质问答 | **强**（2D 图到文本生成） | 局限于静态二维化学图拓扑，**零三维空间坐标、零等变不可约表示、零动力学轨迹感知** |
| 32 | **Coscientist**: *Autonomous chemical research with large language models* — Boiko, MacKnight, Kline & Gomes, **Nature 2023**, 624, 570–578 (`arXiv:2304.05332`) | 自然语言研究目标、在线文档、SMILES 字符串与 Python/实验硬件控制日志 | 自动化实验方案、代码脚本及自然语言反应规划报告 | **强**（工具增强型自主智能体） | 纯文本与外部 API 编排层，对三维原子坐标、`O(3)` 张量场与微观轨迹力学过程完全不可见 |
| 33 | **ChemCrow**: *Augmenting large language models with chemistry tools* — Bran et al., **Nat. Mach. Intell. 2024**, 6, 525–535 (`arXiv:2304.05376`) | 自然语言指令 + 1D SMILES 字符串（调用 18 种外部 RDKit/化学计算 Python 工具） | 合成路线规划、安全评估与分子标量性质自然语言回答 | **强**（ReAct 工具调用智能体） | 仅能处理一维字符串与标量工具返回值，完全缺乏三维等变编码器与动力学轨迹分析能力 |
| 34 | **3D-MoLM**: *Towards 3D Molecule-Text Interpretation in Language Models* — Li et al., **ICLR 2024** (`arXiv:2401.13923`) | **单一静态平衡三维构象** $(Z_i, \mathbf{r}_i)$ 经 Uni-Mol 提取 $SE(3)$ **不变距离表示 (`0e`)** + 3D-Q-Former 投影 | 三维分子几何特征自然语言解读与量子化学标量属性（HOMO-LUMO 等）文本问答 | **强**（静态单构象 3D-to-Text） | **仅支持单帧静态构象且抹除了 `1o/2e` 方向性等变张量**；完全无法摄入多帧 $\Delta\mathbf{r}_i$ (`1o`)、原子力 $\mathbf{F}_i$ 或诊断质子传递动态事件 |
| 35 | **ChatMol Copilot**: *An Agent for Molecular Modeling and Computation Powered by LLMs* — Sun, Li, Deng & Li, **ACL 2024 (LangMol)**, pp. 55–65 | 自然语言建模指令 + PDB/SMILES 文件路径（通过外部 PyMOL / AutoDock CLI 脚本交互） | PyMOL 可视化脚本执行、对接打分日志及自然语言摘要 | **强**（脚本编排型建模助手） | 大模型从未直接接触三维几何隐表示或轨迹张量场，分析上限完全受限于预置 CLI 脚本模板 |
| 36 | **AtomAgents**: *Alloy design and discovery through physics-aware multi-modal multi-agent artificial intelligence* — Ghafarollahi & Buehler, **PNAS 2025**, 122, e2414074122 (`arXiv:2407.10022`, 2024) | 自然语言材料设计任务 + LAMMPS 输入脚本、标量热力学日志与渲染的 2D 曲线图（经 VLM 读取） | 自动化 LAMMPS 模拟执行、多智能体协同合金材料设计与自然语言机理分析报告 | **强**（多模态多智能体系统） | 仅通过文本日志与二维统计截图**间接观察**模拟结果，缺失原子级 `(pos, delta_pos_1o, force_1o)` 到语言的直接隐空间投影 |
| 37 | **MDCrow**: *Automating Molecular Dynamics Workflows with Large Language Models* — Campbell, Cox, Medina, Watterson & White, **arXiv 2025** (`arXiv:2502.09565`) | 自然语言生物分子模拟任务 + PDB 拓扑（调用 OpenMM 与 MDTraj 脚本计算 RMSD、$R_g$、预设氢键数） | 自动运行 MD 工作流并基于脚本输出的标量统计量生成自然语言总结 | **强**（CoT 工具链 MD 智能体） | 典型“外挂脚本包裹器”瓶颈：若发生未预设脚本的局部反应过垒或恢复力异常，LLM 无法从底层几何张量直接感知 |
| 38 | **EquiLLM**: *Large Language-Geometry Model: When LLM meets Equivariance* — Li, Cen, Su, Huang, Xu, Rong & Zhao, **ICML 2025** (`arXiv:2502.11149`) | 三维坐标/轨迹拆分为等变编码器方向分支 (`1o`) 与几何感知标量提示 (`0e` 距离/PCA) 输入 LLM + 等变适配器 | **未来三维坐标/结构数值预测**（MD17 分子轨迹、人体动作、抗体 3D 环区），**非自然语言机理文本** | **受限**（LLM 仅充当内部不变特征算子，不输出物理分析文本） | 模态方向恰好相反（`3D -> LLM -> 3D 坐标` 而非 `动态 Irreps -> 自然语言诊断`），且将 LLM 与 `1o` 力-位移做功耦合隔离，不支持反应/质子转移解释 |
| 39 | **El Agente**: *El Agente: An Autonomous Agent for Quantum Chemistry* — Zou et al. & Aspuru-Guzik, **Matter 2025** (`arXiv:2505.02484`) | 自然语言量子化学计算需求 + ORCA 等电子结构程序输入/输出文本文件与几何坐标文件 | 自主规划执行从头算过渡态搜索、热化学及光谱计算并生成自然语言分析报告 | **强**（分层记忆量子化学智能体） | 通过解析量子化学程序输出日志做符号推理，不具备面向连续 MD 轨迹高阶 `O(3)` 不可约表示的端到端几何感知 |
| 40 | **MDAgent2**: *Large Language Model for Code Generation and Knowledge Q&A in Molecular Dynamics* — Shi et al., **arXiv 2026** (`arXiv:2601.02075`) *(同参 Ma et al., `MDAgent`, `arXiv:2604.18622`, 2026)* | 自然语言分子动力学模拟规格说明与领域知识问题（经 SFT + RL 后训练） | 可执行 LAMMPS/GROMACS 模拟代码生成与领域知识自然语言问答 | **强**（垂域强化微调代码与问答大模型） | 纯 `Text-to-Code` / `Text-to-Text` 架构，无法直接读取实时三维原子轨迹 $(\mathbf{r}, \Delta\mathbf{r}_{1o})$、逐原子受力 $\mathbf{F}_i$ 或输出微观事件诊断 |

---

## 3. 物理与几何对齐准则综合提炼（Synthesis of Alignment Requirements）

将上述 40 篇文献中的群表示论、非平衡统计力学与反应动力学判据加以统一形式化，可提炼出连接**微观轨迹几何张量**与**宏观自然语言物理描述**的三大严格对齐公理：

### 3.1 `O(3)` / `E(3)` 奇宇称矢量（`1o`）的速度/位移注入与张量积完备分解
在三维正交群 $O(3) = SO(3) \times \mathbb{Z}_2$ 下，任何物理量必须严格按照角动量量子数 $l \in \{0, 1, 2, \dots\}$ 与空间反演宇称 $p \in \{+1 (\text{even}, e), -1 (\text{odd}, o)\}$ 分类：
1. **EGNN 的笛卡尔速度等变更新准则（Satorras et al., ICML 2021）**：
   对于携带初始速度（或帧间有限差分位移）$\mathbf{v}_i^{(l)} \in \mathbb{R}^3$ 的节点，其更新必须保持旋转/反射等变与平移不变：
   $$\mathbf{v}_i^{(l+1)} = \phi_v\!\left(\mathbf{h}_i^{(l)}\right)\mathbf{v}_i^{(l)} + C \sum_{j \neq i} \left(\mathbf{x}_i^{(l)} - \mathbf{x}_j^{(l)}\right)\phi_x(\mathbf{m}_{ij}), \qquad \mathbf{x}_i^{(l+1)} = \mathbf{x}_i^{(l)} + \mathbf{v}_i^{(l+1)}$$
   该形式轻量高效，但因缺乏 $l \ge 1$ 之间的完整 CG 张量积，无法区分共线拉伸、剪切摆动与扭转涡度。
2. **SEGNN 的可控球谐 `1o` 物理量注入准则（Brandstetter et al., ICLR 2022）**：
   在周期性边界条件（PBC，晶胞矩阵 $\mathbf{H}$）下，定义服从最小镜像约定（Minimum-Image Convention, MIC）的帧间原子位移矢量：
   $$\Delta \mathbf{r}_i(t) = \operatorname{MIC}\!\big(\mathbf{r}_i(t + \Delta t) - \mathbf{r}_i(t),\, \mathbf{H}\big) \in \mathbf{1o}$$
   原子瞬时位移 $\Delta\mathbf{r}_i$、速度 $\mathbf{v}_i = \Delta\mathbf{r}_i / \Delta t$、相对键矢 $\mathbf{r}_{ij}$ 以及势能负梯度力 $\mathbf{F}_i = -\nabla_{\mathbf{r}_i}E$ 在空间反演 $\mathbf{r} \mapsto -\mathbf{r}$ 下均变号，因此**严格属于奇宇称极矢量不可约表示 `1o`（$l=1, p=-1$）**。按照 `SEGNN` 范式，节点可控属性 $\tilde{\mathbf{a}}_i$ 与边可控属性 $\tilde{\mathbf{a}}_{ij}$ 通过实球谐展开构造，并在消息传递中执行 Clebsch-Gordan 张量积。
3. **`1o`（受力/键轴）$\otimes$ `1o`（位移/速度）的群论黄金分解定理**：
   将奇宇称力矢量 $\mathbf{F}_i \in \mathbf{1o}$ 与奇宇称帧间位移矢量 $\Delta\mathbf{r}_i \in \mathbf{1o}$ 做 $O(3)$ 张量积，根据宇称乘法 $(-1)\times(-1)=+1$ 与角动量耦合规则，其直和分解**唯一且精确地**给出三个偶宇称物理通道：
   $$\mathbf{1o} \otimes_{\text{CG}} \mathbf{1o} = \mathbf{0e} \oplus \mathbf{1e} \oplus \mathbf{2e}$$
   - **`0e` 标量通道（方向性机械功 / 瞬时功率）**：$W_i = \mathbf{F}_i \cdot \Delta\mathbf{r}_i$，严格 $O(3)$ 不变，直接度量势场对该原子做正功（驱动加速）还是负功（恢复力阻尼/攀爬能垒）；
   - **`1e` 赝矢量通道（角动量力矩 / 摆动涡度）**：$\boldsymbol{\tau}_i^{\text{kin}} = \mathbf{F}_i \times \Delta\mathbf{r}_i$（或局部键旋转 $\hat{\mathbf{r}}_{ij} \times \Delta\mathbf{r}_i$），为轴矢量（Axial Vector），精确刻画水分子天平摆动（libration）与氢键重排旋转门控；
   - **`2e` 二阶无迹对称张量通道（微观运动学维里通量 / 局域剪切畸变）**：$\mathbf{Q}_i = \frac{1}{2}\big(\mathbf{F}_i \otimes \Delta\mathbf{r}_i + \Delta\mathbf{r}_i \otimes \mathbf{F}_i\big) - \frac{1}{3}(\mathbf{F}_i \cdot \Delta\mathbf{r}_i)\mathbf{I}$，刻画过渡态溶剂笼的各向异性压缩与张应力重构。

   这一分解从数学上证明：**只要在几何骨干中显式耦合 `1o` 位移与 `1o` 受力/键矢，就能零信息损耗地同时析出机械功（`0e`）、旋转门控（`1e`）与各向异性应力（`2e`）**。

### 3.2 化学反应过渡态与 Grotthuss 质子传递的标准集体变量（CVs）
综合 Cluster C 文献（Calegari Andrade et al., 2020; Bonati et al., `mlcolvar` 2023; Di Pino et al., `ZundEig` 2023; Gomez, Thompson & Laage, *Nat. Chem.* 2024），描述质子传递及共价键交换的核心几何坐标体系包含以下三个互补量：
1. **Grotthuss 质子共享不对称伸缩坐标 $\delta(t)$**：
   对于潜在的供体重原子 $\text{D}$（如 $\text{O}_1$）、受体重原子 $\text{A}$（如 $\text{O}_2$）及桥连活泼质子 $\text{H}^*$:
   $$\delta(t) = d_{\text{MIC}}(\text{D}, \text{H}^*; t) - d_{\text{MIC}}(\text{A}, \text{H}^*; t)$$
   - **定域共价态 / Eigen 态（$\text{H}_9\text{O}_4^+$）**：$|\delta(t)| > 0.25\text{ \AA}$（$\delta < -0.25\text{ \AA}$ 时质子定域于供体 $\text{D}$；$\delta > +0.25\text{ \AA}$ 时完成向受体 $\text{A}$ 的共价转移）；
   - **对称共享过渡态 / Zundel 态（$\text{H}_5\text{O}_2^+$）**：$|\delta(t)| \le 0.10\text{--}0.15\text{ \AA}$，且伴随供-受体重原子间距显著压缩 $R_{\text{DA}}(t) = d_{\text{MIC}}(\text{D}, \text{A}; t) \lesssim 2.40\text{--}2.52\text{ \AA}$（远短于普通液态水氢键平衡距离 $\sim 2.8\text{ \AA}$）。
2. **可微光滑配位数（Smooth Coordination Number, `mlcolvar` / PLUMED 标准有理开关函数）**：
   硬截断阶跃函数在截断半径处不可微且易受热噪声干扰，文献统一采用光滑有理开关函数追踪重原子 $i$ 的瞬时质子配位数 $C_i(t)$：
   $$C_i(t) = \sum_{j \in \mathcal{H}} \frac{1 - \big(r_{ij}^{\text{MIC}}(t) / r_0\big)^n}{1 - \big(r_{ij}^{\text{MIC}}(t) / r_0\big)^m}, \qquad (n=6,\; m=12,\; r_0 \approx 1.30\text{ \AA}\text{ 对 O--H 键})$$
   中性水分子满足 $C_{\text{O}} \approx 2.0$，水合氢离子核满足 $C_{\text{O}} \approx 3.0$，氢氧根满足 $C_{\text{O}} \approx 1.0$；有效质子跳跃对应供体 $\Delta C_{\text{D}} \approx -1$ 与受体 $\Delta C_{\text{A}} \approx +1$ 的反相协同跃迁。
3. **非往返重穿（Non-Rattling）氢键门控判据（Gomez, Thompson & Laage, *Nat. Chem.* 2024）**：
   单纯瞬时跨越 $\delta = 0$ 有大量概率属于皮秒内的瞬态往返振荡（rattling recrossing）。真正的不可逆反应跃迁要求 $\delta(t)$ 发生符号翻转（$\delta(t_-)\cdot \delta(t_+) < 0$）并在短驻留窗口内稳定进入新吸引盆（$|\delta(t_+)| > \delta_{\text{commit}}$），同时伴随第一溶剂壳层配位氢键的重整。

### 3.3 逐原子方向性机械功 $W_i = \mathbf{F}_i \cdot \Delta\mathbf{r}_i$ 与能垒攀爬/弛豫力学判据
传统轨迹脚本仅监控标量势能 $E(t)$ 或力模长 $\|\mathbf{F}_i\|$，在大体系中极易被背景溶剂热涨落（$\sqrt{N} k_B T$）淹没。引入逐原子机械功 $W_i(t) = \mathbf{F}_i(t) \cdot \Delta\mathbf{r}_i(t)$ 及方向余弦 $\cos\theta_i(t) = \frac{\mathbf{F}_i(t)\cdot\Delta\mathbf{r}_i(t)}{\|\mathbf{F}_i(t)\|\|\Delta\mathbf{r}_i(t)\| + \varepsilon}$ 可建立局域能量转换的显微镜：
- **能垒攀爬 / 恢复力做负功区（Up-Barrier / Restoring Regime，$W_i < 0,\; \cos\theta_i < 0$）**：原子依靠局域动能或溶剂笼挤压**逆着保守势场恢复力方向推进**，动能转化为局域化学势能（$\Delta V_i \approx -W_i > 0$），对应断键拉伸或 $\text{D}\cdots\text{A}$ 溶剂笼主动压缩接近鞍点的阶段；
- **一阶鞍点 / 准过渡态拐点（Saddle Inflection，$|\delta| \to 0,\; \mathbf{F}_i \cdot \hat{\mathbf{e}}_{\text{DA}} \approx 0$ 但 $\Delta\mathbf{r}_i \cdot \hat{\mathbf{e}}_{\text{DA}} \neq 0$）**：沿反应坐标方向的恢复力过零变号，而质子沿供-受体轴仍具非零前向速度，精准区分**真实反应鞍点穿越**与高排斥力硬球热碰撞；
- **下坡驱动 / 振动弛豫做正功区（Downhill Driving / Relaxation Regime，$W_i > 0,\; \cos\theta_i > 0$）**：越过鞍点后，新共价键吸引力顺着位移方向做正功，释放反应自由能并耗散至周围氢键网络。

---

## 4. 原创性与学术空白区论证（Creativity & Whitespace Analysis）

将 YLM 的**“基于 `(pos, delta_pos_1o, force_1o)` + 方向性机械功/恢复力分解 + 自然语言事件引导 `IrrepsToText` 查询”**置于上述 40 篇文献构成的坐标系中，其原创性体现在对三大现有范式结构性盲区的正交突破：

| 对比维度 | 沉默型等变力场（NequIP / MACE / Allegro / ViSNet） | 速度/生成式轨迹网络（SEGNN / Timewarp / MDGen / EquiJump） | 脚本调用型 MD 智能体 & 静态 3D-LLM（MDCrow / AtomAgents / MDAgent2 / 3D-MoLM） | 内嵌 LLM 的等变坐标预测器（EquiLLM, ICML 2025） | **YLM 事件引导动态 `IrrepsToText`（本工作定位）** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PBC 安全的高阶 `O(3)` 几何等变性** | ✅ (`0e..3o`) | ✅ / 部分 | ❌（`3D-MoLM` 仅静态 `0e`） | ✅ | **✅（完整保留奇偶宇称 irreps）** |
| **帧间位移/速度 `1o` 张量耦合** | ❌（仅单帧静态 $\mathbf{r}$） | ✅（数值隐状态） | ❌ | ✅（仅服务坐标预测） | **✅（$\mathbf{1o}\otimes\mathbf{1o}\to \mathbf{0e}\oplus\mathbf{1e}\oplus\mathbf{2e}$）** |
| **逐原子方向性机械功 $W_i = \mathbf{F}_i\cdot\Delta\mathbf{r}_i$** | ❌ | ❌ | ❌ | ❌ | **✅（精确区分恢复力攀爬 vs 驱动弛豫）** |
| **反应/质子转移精确 CV（$\delta, C_i$）融合** | ❌（需外挂 PLUMED） | ❌（`GMN/MDGen` 锁定共价拓扑） | ⚠️（仅限写死外部脚本） | ❌ | **✅（内生几何-力学联合门控）** |
| **面向科学家的自然语言机理输出** | ❌ | ❌ | ✅（但脱离底层张量场） | ❌（输出为 3D 坐标） | **✅（对称性自洽的 `IrrepsToText` 诊断）** |

**为什么这一组合具有不可替代的学术原创性？**
1. **超越“沉默型”等变势（NequIP/MACE）的信息坍缩瓶颈**：`NequIP` 与 `MACE` 虽然在隐层构造了精妙的 `1o/2e` 等变张量，但为了保证能量标量性，在读出头将高阶通道全部缩并或丢弃，且从未将连续两帧的位移矢量 $\Delta\mathbf{r}_i$ (`1o`) 与受力 $\mathbf{F}_i$ (`1o`) 做张量耦合。YLM 将这些原本在力场读出前被丢弃的高阶动态不可约表示直接转化为可被语言查询的物理语义 Token，实现了从“只会算力”到“能解释力学因果”的跃迁。
2. **超越“脚本包裹型”MD 智能体（MDCrow/AtomAgents/MDAgent2）的“盲人摸象”困境**：现有 MD 智能体本质上是“代码生成器 + 日志阅读器”，大模型只能看到后处理脚本打印出的全局平均数值。若用户追问*“在第 140–180 fs 窗口内，哪一对氧原子之间发生了受阻的质子试探性攀爬（rattling）而非真正转移？周围溶剂笼施加的是恢复阻力还是推力？”*，未预写专门脚本的智能体要么报错，要么产生严重幻觉。而 YLM 直接在隐空间与物理槽位上编码 `(pos, delta_pos_1o, force_1o)` 与 $(\delta, W_i)$，无需人工临时拼凑解析脚本即可按自然语言查询精准定位微观事件。
3. **与最新前沿 `3D-MoLM`（ICLR 2024）及 `EquiLLM`（ICML 2025）形成本质区隔**：`3D-MoLM` 只能描述单帧平衡态分子长什么样（且丢失了方向性 `1o/2e` 信息）；`EquiLLM` 则将 LLM 用作预测未来三维坐标的内部不变算子，根本不产生自然语言解释。**YLM 首次打通了“动态多帧 `O(3)` 不可约表示 + 逐原子机械功分解 + 反应过渡态集体变量 $\longrightarrow$ 旋转平移不变的自然语言物理因果报告”这一正向科学发现闭环**。

---

## 5. 面向 YLM 分子动力学模块（`src/toolkit/domains.py`）的 5 项具体架构升级建议

基于上述 40 篇文献验证的数学与物理机制，建议对 YLM 的 MD Trajectory 领域模块（`src/toolkit/domains.py`）实施以下 **5 项可严格验证的架构与物理特征升级**：

### 升级 1：引入 SEGNN 风格的 PBC 安全 `1o` 位移/速度通道与 $\mathbf{1o} \otimes \mathbf{1o} = \mathbf{0e} \oplus \mathbf{1e} \oplus \mathbf{2e}$ 张量耦合
- **文献依据**：`SEGNN`（Brandstetter et al., ICLR 2022, No. 12）、`EGNN`（Satorras et al., ICML 2021, No. 11）。
- **设计落地**：
  1. 对相邻轨迹帧计算严格服从周期性晶胞最小镜像约定（MIC）的位移矢量 $\Delta\mathbf{r}_i(t) = \operatorname{MIC}(\mathbf{r}_i(t+\Delta t) - \mathbf{r}_i(t), \mathbf{H})$，将其显式标记为奇宇称球谐不可约表示 `1o`（而非当作普通标量坐标差拼接，避免破坏旋转与空间反演对称性）。
  2. 在节点/边等变特征构建中，显式计算受力/键矢（`1o`）与帧间位移（`1o`）的 CG 分解三分量：标量内积功 `0e`（$\mathbf{F}_i \cdot \Delta\mathbf{r}_i$）、轴矢量力矩/摆动涡度 `1e`（$\mathbf{F}_i \times \Delta\mathbf{r}_i$ 与 $\hat{\mathbf{r}}_{ij} \times \Delta\mathbf{r}_i$）以及二阶无迹对称运动学维里张量 `2e`，使 `IrrepsToText` 同时感知平动做功、氢键旋转摆动与局域剪切畸变。

### 升级 2：内置精确 Grotthuss 质子共享坐标 $\delta = d_1 - d_2$、可微光滑配位数 $C_i$ 与非往返门控判据
- **文献依据**：Calegari Andrade et al.（Chem. Sci. 2020, No. 21）、`mlcolvar`（Bonati et al., JCP 2023, No. 25）、`ZundEig`（Di Pino et al., JPCB 2023, No. 27）、Gomez, Thompson & Laage（Nat. Chem. 2024, No. 29）。
- **设计落地**：
  1. 在 MD 轨迹特征提取器中自动识别潜在桥连氢原子 $\text{H}^*$ 及其最近邻/次近邻重原子受体对 $(\text{D}, \text{A})$（支持 $\text{O, N, S, F}$ 等杂原子），实时计算 MIC 修正下的**质子共享坐标 $\delta(t) = d(\text{D--H}^*) - d(\text{A--H}^*)$** 与**供-受体压缩距离 $R_{\text{DA}}(t) = d(\text{D--A})$**。
  2. 引入 `mlcolvar` 标准光滑有理配位数 $C_i(t) = \sum_{j\in\mathcal{H}} \frac{1-(r_{ij}/r_0)^6}{1-(r_{ij}/r_0)^{12}}$，避免硬截断键长阈值在过渡态附近引发数值跳变伪影。
  3. 增设双重门控状态标签：`Eigen/Localized`（$|\delta| > 0.25\text{ \AA}$）、`Zundel/Shared-TS`（$|\delta| \le 0.12\text{ \AA}$ 且 $R_{\text{DA}} \le 2.52\text{ \AA}$）、`Committed-Transfer`（$\delta$ 跨零变号且伴随 $\Delta C_{\text{D}} < -0.5, \Delta C_{\text{A}} > +0.5$ 持续驻留）与 `Frustrated-Rattling`（瞬时 $|\delta| \le 0.15\text{ \AA}$ 但旋即退回原供体盆），从根源上消除将热涨落往返误判为成功反应的假阳性。

### 升级 3：逐原子方向性机械功 $W_i = \mathbf{F}_i \cdot \Delta\mathbf{r}_i$ 与“恢复力攀爬 vs 驱动弛豫”力学相分解
- **文献依据**：`Transition1x`（Schreiner et al., Sci. Data 2022, No. 23）、Jung & Hummer（Nat. Comput. Sci. 2023, No. 24）。
- **设计落地**：
  1. 计算每个原子及反应活性中心在每一步的方向性机械功 $W_i(t) = \mathbf{F}_i(t) \cdot \Delta\mathbf{r}_i(t)$、累积窗口功 $\mathcal{W}_i^{[t_1, t_2]} = \sum_{t=t_1}^{t_2} W_i(t)$ 以及力-位移夹角余弦 $\cos\theta_i(t)$。
  2. 将轨迹局部动力学严格划分为三个可解释物理相并注入语义摘要槽：
     - **Restoring / Barrier-Climbing（逆梯度攀爬相）**：$W_i(t) < 0$ 且 $\cos\theta_i(t) < -0.3$，表明原子正克服恢复力向高势能区推进；
     - **Saddle-Point Crossing（鞍点穿越相）**：活性质子 $|\delta(t)| \to 0$、沿供-受体轴投影恢复力 $|\mathbf{F}_{\text{H}^*}\cdot\hat{\mathbf{e}}_{\text{DA}}| \to \min$ 而前向位移 $\Delta\mathbf{r}_{\text{H}^*}\cdot\hat{\mathbf{e}}_{\text{DA}} > 0$；
     - **Driving / Downhill-Relaxation（顺梯度驱动弛豫相）**：$W_i(t) > 0$ 且 $\cos\theta_i(t) > +0.3$，表明成键吸引力正主动加速原子完成构象弛豫。

### 升级 4：事件引导的多时间尺度稀疏关键帧提取（避免时间均值池化淹没亚皮秒稀有事件）
- **文献依据**：`ITO`（Schreiner et al., NeurIPS 2023, No. 17）、`MDGen`（Jing et al., 2024, No. 19）、Jung et al.（Nat. Comput. Sci. 2023, No. 24）。
- **设计落地**：
  在长轨迹中，Grotthuss 质子跨越 Zundel 鞍点仅持续 $10\text{--}50\text{ fs}$，若对全轨迹 $T$ 帧做均匀时间均值池化，99% 的近平衡热振动会将关键过垒帧的 irreps 信号彻底稀释。建议在 `domains.py` 中实现**事件引导关键帧选择器（Event-Guided Keyframe Selector）**：依据复合显著性得分 $S(t) = \alpha \exp(-|\delta(t)|/\sigma_\delta) + \beta \max_i |W_i(t)| + \gamma \max_i |\Delta C_i(t)|$ 自动锚定峰值帧 $t^*$ 及其前后因果上下文窗口 $[t^* - \Delta\tau,\, t^* + \Delta\tau]$（分别对应 Pre-TS 笼压缩帧、TS 鞍点帧、Post-TS 弛豫帧），以三阶段时序锚点喂入 `IrrepsToText`。

### 升级 5：严格保持 $SE(3)$ 语义不变性的物理锚定 `IrrepsToText` 投影契约
- **文献依据**：对比修正 `3D-MoLM`（Li et al., ICLR 2024, No. 34）与 `EquiLLM`（Li et al., ICML 2025, No. 38）的模态错位。
- **设计落地**：
  自然语言文本本身必须是全局刚体旋转与平移不变的（即不能依赖任意选取的实验室笛卡尔坐标轴 $x/y/z$ 报告方向，否则旋转晶胞后答案会自相矛盾），但必须保留**相对内禀方向性**（如“质子 $\text{H}_{12}$ 沿 $\text{O}_3 \to \text{O}_7$ 氢键轴移动，并受到来自受体溶剂壳层的瞬时恢复阻力”）。因此，在 `IrrepsToText` 读出接口中，应将高阶等变通道的局部范数谱（$\|\mathbf{h}^{(0e)}\|, \|\mathbf{h}^{(1o)}\|, \|\mathbf{h}^{(1e)}\|, \|\mathbf{h}^{(2e)}\|$）与内禀相对投影标量（$\delta(t^*), R_{\text{DA}}(t^*), W_i(t^*), \cos\theta_i(t^*), \Delta C_i$）联合绑定为结构化条件特征，既从群论上严格保证旋转/平移/宇称不变的语言输出，又彻底杜绝脚本型智能体的数值幻觉。
