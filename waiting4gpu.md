# `waiting4gpu.md` — 拿到 GPU 后的完整作战手册（YLM Equivariant Foundation Model & Live MD Hooks）

> **状态**：核心架构、等变算子优化、周期性晶胞（PBC）、二阶张量读出头（`0e+2e`）、`MACEEncoder` 底座适配器以及 OpenMM / LAMMPS 实时反应监测绘图插件**均已在 CPU 上开发完毕并通过全部单元测试**。只等 GPU 到位即可直接启动大规模 DFT 训练与打榜实验。

---

## 0. 先回答：现在没有 GPU，哪些能做、哪些必须等 GPU？

| 类别 | 没有 GPU 能不能做？ | 说明 |
| :--- | :---: | :--- |
| **跑通 `ylm` 命令行、3D 网页报告、Web Studio (`ylm serve`)** | **✅ 现在就能做** | 默认模型仅 `1.7 MB`，纯 CPU 单分子推理约 `30 ms`。 |
| **挂载 `MACE-MP-0-small` / `MACE-OFF23-small` 测单分子或十几帧短轨迹** | **✅ 现在就能做** | CPU 上单帧约 `0.2–0.8 s`，足够验证代码链路、测试 OpenMM/LAMMPS 插件和生成时间轴反应图。 |
| **提前写好/跑通数据预处理脚本（500 条小样本 Smoke Test）** | **✅ 强烈建议无 GPU 时先做** | 把数据格式转换、子结构文本生成、Loss 下降曲线在 CPU 上用 100 条样本跑通 5 步，避免拿到宝贵 GPU 后把时间浪费在修路径和字段 Bug 上。 |
| **百万级 DFT 预训练 + 正面对比 `Uni-Mol` / `GraphMVP` / `3D-InfoMax`** | **⏳ 必须等 GPU** | 涉及百万构象球谐张量积与多层消息传递，CPU 会慢 50–100 倍，必须上 GPU。 |
| **长时程可反应 MLIP 分子动力学（几十~几百 ps 真实反应采样）** | **⏳ 必须等 GPU** | OpenMM-Torch + MACE 跑几万步 MD 需要单张 GPU 加速。 |

---

## 1. 项目的两条核心主线（北极星目标）

### 主线一：等变多模态 3D 科学基座模型（`YLM-FM`，对标并超越 `Uni-Mol` / `GraphMVP` / `3D-InfoMax` / `3D-MoLM`）
- **核心痛点**：
  - `GraphMVP` / `3D-InfoMax` 把 3D 构象蒸馏进 2D 图，推理时完全丢失三维方向；
  - `Uni-Mol` / `Uni-Mol2` 只吃原子间距离矩阵（纯 `0e` 标量），原理上无法输出偶极方向 $\boldsymbol{\mu}$、局部受力矢量 $\mathbf{F}_i$、四极/极化率/介电二阶张量（`0e+2e`）；
  - `3D-MoLM` / `EquiLLM` 在进入 LLM 时把高阶方向压扁成了标量，无法用自然语言对局部原子和化学键做跨原子张量组合（如力差值 $\mathbf{F}_A - \mathbf{F}_B$、键轴投影、各向异性极化）。
- **我们的破局方案**：
  - 底层直接挂载预训练全元素物理底座 **`MACE-MP-0`（89 种元素晶体/材料） / `MACE-OFF23`（有机/药物分子）**（代码已就绪：`src/mace_backbone.py`）；
  - 中间用 **`IrrepsToText`（Option B + `channelwise` Option C + `--n_cross_layers`）** 保留完整 `0e + 1o + 2e` 等变流；
  - 顶层同时支持**不变标量任务（能量、HOMO-LUMO、溶解度、电导率、结合亲和力）** + **等变向量任务（偶极、局部力、过渡态本征方向）** + **二阶张量任务（四极矩、极化率、介电/应力张量）** + **自然语言子结构寻址与生成（`<vec>`）**。

### 主线二：面向 MLIP 分子动力学的实时反应监测插件（OpenMM / LAMMPS Add-on）
- **核心功能**：在跑可断键/可反应力场（OpenMM-Torch + MACE/ANI/NequIP，或 LAMMPS ReaxFF / `pair_style mace`）时挂载 `OpenMMReactionReporter` 或 `LAMMPSReactionHook`（代码已就绪：`src/toolkit/md_hooks.py`）。
- **运行机制**：平时零 LLM 显存开销，用带双阈值迟滞的光滑键级矩阵 $B_{ij}(t)$ 与 Grotthuss 质子共享坐标 $\delta(t)$ 实时扫描；一旦发生**成键、断键、质子转移**，自动记录帧号与飞秒/皮秒时间戳，计算局部机械功 $W_i = \mathbf{F}_i \cdot \Delta\mathbf{r}_i$（区分顺推弛豫 vs. 逆推爬垒），实时输出 JSON 事件表、时间轴折线图（`.svg` / `.png`）与 3D 双箭头对比网页。

---

## 2. 当前仓库已就绪的轮子清单（拿到 GPU 后直接调用，无需重写）

| 模块路径 | 已实现的核心能力 |
| :--- | :--- |
| `src/mace_backbone.py` (`MACEEncoder`) | 一键加载 `mace-mp-0-{small,medium,large}` 或 `mace-off23-{small,medium,large}`，自动做 Bohr $\leftrightarrow$ Å 转换、支持分子与 3D PBC 晶胞、拼接 `node_feats` 与精确保守力梯度 `1x1o`。 |
| `src/model.py` (`YLM`, `YLM.from_mace`) | 支持 Option A/B/C、`--c_tp_mode channelwise`（参数量降 `21x`、提速 `10x`）、多层交互 `--n_cross_layers`、向量读出 `1x1o` 与二阶对称张量读出 `tensor_readout=True` (`0e+2e`)。 |
| `src/ylm_lm.py` (`YLMCausalLM`) | 支持 `hf:Qwen/...` 等开源大模型中间层挂载 `IrrepsAdapter`、`<vec>` 三维向量生成读出、帧间运动向量 `use_motion_1o=True` 及 `freeze_encoder=True`。 |
| `src/encoder.py` (`_periodic_edges`) | 支持 1D/2D/3D 周期性边界条件（含小晶胞周期性自镜像边）与帧间最小镜像位移。 |
| `src/baselines.py` | 已内置公平对比基线：`EquiLLMBaseline`（仅不变特征进 Transformer）与 `FrameAveragingBaseline`（8 标架 PCA 平均）。 |
| `src/toolkit/md_hooks.py` | 已内置 `LiveReactionMonitor`、`OpenMMReactionReporter`、`LAMMPSReactionHook` 与 `plot_reaction_timeline`。 |
| `src/toolkit/hf_export.py` | 一键导出 `TensorQA` 多配置 Hugging Face 数据集（含 `--anonymized` 双盲匿名模式与 $K=32$ 旋转评测脚本）。 |

---

## 3. 按 GPU 算力档位的对号入座指南

拿到 GPU 后，先看显存和卡数选对应档位：

- **档位 1：单卡 16–24 GB（Colab T4 / L4 / 单张 RTX 3090 / 4090）**
  - **推荐配置**：**冻结 `MACE-OFF23-small` 或 `MACE-MP-0-small`（`freeze_mace=True`）** + 训练 `YLM`（`option="C"`, `c_tp_mode="channelwise"`, `n_cross_layers=2`）+ 冻结 `MatSciBERT`。
  - **显存占用**：`< 6 GB`（Batch Size 64–128），单卡一天即可跑完 `SPICE-v2` 20 万–50 万样本或完整 `MPtrj` 精简集，足够产出扎实的第一篇方法+应用论文结果！
- **档位 2：单卡 40–80 GB（Colab A100 / 学校集群 Hoffman2 / Expanse 单卡 A100/H100）**
  - **推荐配置**：挂载 **`MACE-OFF23-medium` / `MACE-MP-0-medium`（含 `L=1` 等变隐层 + 保守力 `1x1o`）**，前半程冻结 MACE 热启动跨模态层，后半程小学习率（`1e-5`）联合微调顶层 interaction + `YLMCausalLM`（`hf:Qwen/Qwen2.5-1.5B` LoRA + `<vec>`）。
  - **可跑规模**：完整 `SPICE-v2`（200 万构象）+ `MPtrj`（150 万帧）+ `Transition1x` 反应路径子集。
- **档位 3：多卡 4–8× A100 / H100（冲刺全化学宇宙 `YLM-Universe`）**
  - **推荐配置**：`MACE-MP-0-large`（`L=2` 完整 `0e+1o+2e` 89 元素底座），混合采样 `OMol25`（83 元素 DFT）+ `OMat24` + SES AI `MU` + `Transition1x`，开启多泛函条件标签。

---

## 4. 数据集清单与“物理方向 vs. 绝对值大小”Universe 训练配方

### 4.1 数据集分工表

| 数据集 | 体系范围与规模 | 提供的量子化学标签 | 训练分工（训方向 vs. 训绝对值） |
| :--- | :--- | :--- | :--- |
| **`SPICE-v2`** | 200 万构象（类药分子、二肽、水合 `Li+/Na+/Mg2+/Cl-`） | `ωB97M-D3BJ/def2-TZVPPD`：能量(`0e`)、力(`1o`)、分子偶极(`1o`)、**MBIS 原子偶极(`1o`)与原子四极(`2e`)** | **有机/药物/溶剂化簇的【方向 + 绝对值】双重黄金标尺** |
| **`OMol25`** (Meta) | ~1 亿构象，覆盖 **83 种元素**（电解液、配合物、生物片段） | `ωB97M-V/def2-TZVPD`：能量(`0e`)、力(`1o`)、偶极(`1o`)、电荷/自旋 | **扩展到全元素分子空间的终极 Universe 库**（单卡可先取 100 万分层采样集） |
| **SES AI `MU` / `LIBE` / `OEDB`** | 锂电碳酸酯/醚类溶剂、锂盐阴离子、SEI 自由基与配方 | DFT 能量/HOMO-LUMO(`0e`)、偶极(`1o`)、部分极化率(`2e`) + 宏观电导率 | **电池电解液垂直领域的方向与下游标量评测库** |
| **`MPtrj` + MP Dielectric** | 150 万帧无机晶体（**89 种元素**）+ 数千介电/压电晶体 | `PBE/PBE+U` 原子力(`1o`)、晶胞应力(`0e+2e`)、介电张量(`0e+2e`) | **周期性无机材料与固态电解质的【方向 + 晶体绝对值】标尺** |
| **`Transition1x`** | 1,000 万帧有机化学反应 NEB 过渡态路径 | `ωB97x/6-31G(d)` 反应鞍点恢复力(`1o`)与能垒(`0e`) | **专门把断键/成键/质子转移过渡态附近的【力方向】训准** |
| **`QM7-X` / `Wilkins QM7b`** | 420 万非平衡构象 / 7,211 分子 CCSD 张量 | 分子偶极(`1o`)、原子力(`1o`)、**完整二阶极化率张量(`0e+2e`)** | **二阶各向异性极化张量的高精度基准** |

### 4.2 关键训练技巧：防止多数据集混训把“绝对值”训糊
1. **方向与模长解耦 Loss**：
   $$\mathcal{L}_{\text{vec}} = \lambda_{\text{dir}}\bigl(1 - \cos(\hat{\mathbf{v}}, \mathbf{v}^*)\bigr) + \lambda_{\text{mag}}\,\bigl|\|\hat{\mathbf{v}}\| - \|\mathbf{v}^*\|\bigr| + \lambda_{\text{huber}}\,\|\hat{\mathbf{v}} - \mathbf{v}^*\|_{\text{Huber}}$$
2. **泛函条件标签（Level-of-Theory Token）**：
   不同泛函算出的物理方向高度一致（`cos > 0.995`），但绝对值有 5%–15% 标尺偏移。混训时在文本前缀加上 `[LOT=wb97m]`、`[LOT=pbe_u]` 或 `[LOT=xtb]`（半经验数据只开 $\lambda_{\text{dir}}$，关掉 $\lambda_{\text{mag}}=0$），让所有数据合力把**物理方向**推到 `> 0.995`，同时保证 DFT 绝对值标尺纯净不串味。
3. **解决原子寻址与新句式泛化短板**：
   - 开启显式路由交叉熵损失（`--attn_loss_weight 0.2`），解决 `-OH` 氢原子注意力飘到邻近氧原子的问题；
   - 用 RDKit SMARTS + Pymatgen 配位环境自动生成子结构描述，并用开源 LLM 对每种化学环境扩增 20+ 种自然语言问法，彻底消灭 T4/T5 组合题在新句式上的过拟合。

---

## 5. 拿到 GPU 后的分步执行清单（Step-by-Step Playbook）

### Step 1：GPU 环境一键初始化与冒烟测试（耗时 ~10 分钟）
```bash
git clone https://github.com/yaniguan/YLM.git && cd YLM
git checkout 001-ylm-probe-toolkit   # 若 PR #27 已合并则直接用 main
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[data,crystals,mace,md,test]"

# 1) 跑通全套单元测试确认环境无误
pytest -q

# 2) 验证真实 MACE-MP-0 在 GPU 上挂载 YLM 并通过 16 次随机旋转+镜像反演检验
ylm ask examples/li_ec_shell.xyz "What is the force on the lithium ion?" \
  --mace mace-mp-0-small --device cuda --verify-o3 --html gpu_mace_check.html
```

### Step 2：第一组破局实验 —— 验证 `MACE + YLM` 让物理方向与绝对值双双跃迁（耗时 ~2–4 小时）
在 `SPICE-v2`（或重算 DFT 版 `TensorQA-DFT`）上跑 3 组对照：
1. 原生 3 层从头训 `YLM`（Option C `channelwise`）
2. **冻结 `MACE-OFF23-medium` + `YLM`（Option C `channelwise`, `n_cross_layers=2`）**
3. 微调顶层 `MACE-OFF23-medium` + `YLM`
- **预期里程碑**：单原子力方向余弦从 `v0` 的 `0.960` 跃升至 **`> 0.995`**（角度误差 `< 5°`），绝对值 MAE 进入 **`< 25 meV/Å`** 化学可用区间，且覆盖 `H, C, N, O, F, P, S, Cl, Br, I` 全部药化元素（换 `MACE-MP-0` 覆盖 89 元素）。

### Step 3：正面打榜 `Uni-Mol` / `GraphMVP` / `3D-InfoMax` / `3D-MoLM`（核心论文实验表）

拿到 GPU 后按以下两张表填数：

#### 表 A：传统不变标量基准对比（证明引入三维等变极化场能反哺标量表征）
- **评测集**：QM9 标量（HOMO/LUMO/Gap）、MoleculeNet（ESOL, FreeSolv, Lipophilicity, BACE, BBBP）、电池下游（`CALiSol-23` 电导率、`RedDB` 氧化还原电位、`MP Battery` 电压，`scripts/22_downstream_eval.py --mp_geom_mode pbc`）。
- **对比对手**：`GraphMVP`、`3D-InfoMax`、`Uni-Mol`、`MolCLR`、`EquiLLM`（`src/baselines.py`）、**`YLM-FM (Ours)`**。

#### 表 B：三维方向向量与二阶张量问答基准（降维打击，对手结构性失明）
- **评测集**：`TensorQA-Universe`（T1 偶极矢量、T2 `0e+2e` 四极/极化率/介电张量、T3 语言寻址局部原子力、T4 跨原子力差值、T5 化学键轴向投影，全部在 $K=32$ 随机 $O(3)$ 旋转+反演下评测）。
- **对比对手**：
  - `Uni-Mol` / `3D-MoLM`（距离矩阵标量瓶颈 $\to$ 方向余弦 $\approx 0$）
  - `Qwen2.5` / `Llama-3` 坐标转文本微调（旋转后崩溃，`eq_err > 1.0`）
  - `PCA Frame Averaging`（对称分子简并失稳，`src/baselines.py`）
  - `EquiLLM`（无中间层张量积，T2 四极张量与 T4/T5 组合推理受限）
  - **`YLM-FM (MACE + IrrepsToText Option C channelwise, Ours)`**

### Step 4：OpenMM / LAMMPS 实时反应 MD Showcase 实验（耗时 ~1–2 小时）
用 `openmm-torch` 挂载 `MACE-OFF23` / `MACE-MP-0` 跑两个经典反应体系（各跑 `20–50 ps`），直接挂上 `OpenMMReactionReporter`：
1. **水合质子簇 $\text{H}_5\text{O}_2^+ / \text{H}_9\text{O}_4^+$ Grotthuss 质子传递 MD**：自动输出每次质子跳迁的帧号、皮秒时间戳、Zundel 过渡态 $\delta(t)=0$ 时刻的“运动位移 $\Delta\mathbf{r}$ vs. 恢复力 $\mathbf{F}$”做功曲线图（`grotthuss_timeline.png`）与 3D 交互网页；
2. **`Li+` 在碳酸乙烯酯/氟代碳酸乙烯酯（EC/FEC）混合电解液中的脱溶剂化跳壳 MD**：自动捕捉配位键断裂/新溶剂成键时刻与 `Li+` 逃逸通道向量。

### Step 5：整理权重与 `TensorQA` 数据集上传 Hugging Face
```bash
# 双盲审稿期使用 --anonymized 发布到匿名组织；论文公开后去掉 --anonymized 发布到实名主页
ylm export-hf \
  --labels data/qm9_xtb.pkl \
  --v0 data/tensorqa_v0.json \
  --comp data/tensorqa_comp.json \
  --quad data/qm9_quad.pkl \
  --out dist/hf_tensorqa \
  --anonymized
```
