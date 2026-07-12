# 高维 PDE 深度学习求解 × 张量列方法对比
## 项目设计方案 + 结构化 Prompt 集（v1.0）

> 面向：中科大"一〇七杯"算力平台赛道 · 学科应用方向
> 背景：Sorbonne M2 Mathematics of Modeling，SDEDP（数据科学与 PDE）主修
> 关联课程：MAM84（张量方法求解高维 PDE）、MAM86（神经网络与自适应数值逼近）、MAM95（降维）、MAM35（概率数值方法）

---

# Part A · 项目设计方案

## A0 项目定位（一段话讲清楚）

经典网格法求解 PDE 的成本随维数 d 指数增长（维数灾难）。本项目在同一组高维基准方程（d = 5 → 100）上，**正面对比两类打破维数灾难的方法**：基于随机表示的深度学习方法（Deep BSDE，E–Han–Jentzen）与基于低秩结构的张量列方法（Tensor-Train），系统回答"**误差与成本如何随维数标度、各自的适用边界在哪里**"。所有实验在中科大 107 算力平台上完成，充分利用 CPU 分区、RTX 5090 分区与 A100 分区的异构特性，并以数组作业、断点续算、REST API 监控等方式展示平台工程能力。

**竞赛叙事双主线**：数学深度（两类方法的原理性对比 + 标度律）× 平台能力（异构资源调度 + 大规模扫参 + 可复现工程）。

## A1 科学问题（Research Questions）

| # | 问题 | 对应产出 |
|---|------|---------|
| RQ1 | 固定目标精度下，Deep BSDE 与 TT 的**误差—维数标度**各是什么？ | 误差 vs d 曲线（log-log） |
| RQ2 | 同等精度下谁更便宜？**成本—精度 Pareto 前沿**随 d 如何移动？ | Pareto 图（wall-clock / 显存 vs 相对误差） |
| RQ3 | 解的**低秩结构假设**何时成立？TT 秩随 d 与方程非线性强度如何增长？NN 参数量呢？ | TT 秩–精度–维数三维关系图 |
| RQ4 | **精度与硬件匹配**：fp32/TF32（RTX 5090）够用吗？何时必须 fp64（A100）？ | 精度消融表 + 两分区吞吐对比 |
| RQ5（选做） | Deep BSDE 训练的**种子方差与失败模式**：哪些方程/维度下不稳定？ | 多种子箱线图 + 失败案例分析 |

> 主管备注：RQ1–RQ3 是论文级问题，RQ4 是竞赛加分项（直接展示你理解硬件），RQ5 时间不够就砍。

## A2 方法与基准方程

### 方法栈

**M1 · Deep BSDE（主方法，必做）**
半线性抛物 PDE
`∂u/∂t + ½ Tr(σσᵀ Hess u) + μ·∇u + f(t, x, u, σᵀ∇u) = 0, u(T,·) = g`
经非线性 Feynman–Kac 化为 FBSDE，沿模拟路径把 `u(0, x₀)` 与各时间步的 `σᵀ∇u(tₙ, Xₙ)` 参数化为可训练对象，以终端条件失配为损失。
两个变体（做消融）：
- M1a 原版（E–Han–Jentzen 2018）：每个时间步一个子网络；
- M1b 共享网络变体（FBSNN 风格）：单个网络输入 (t, x)，参数量与 d 解耦更好。

**M2 · 张量列方法（对照方法，分三层降险）**
- **Tier 1（必做）· TT 表征研究**：对有解析解的方程，用 TT-cross（`teneva` 或 `torchtt`）直接逼近解函数 u(0, ·)，测"给定精度所需 TT 秩"随 d 的增长。不需要求解器，纯 CPU，工程量小，但回答 RQ3 的核心。
- **Tier 2（应做）· TT 抛物求解器**：接入 Richter–Sallandt–Nüsken《Solving high-dimensional parabolic PDEs using the tensor train format》（ICML 2021）的公开实现，在同一批方程上与 Deep BSDE 正面对比。
- **Tier 3（选做）· 自实现 ALS/回归版 TT 求解器**：仅当前两层顺利且时间富余。

**M3 · 参考解（裁判，必须可信）**
- 解析解：E1、E2、E5（任意维闭式解）；
- Cole–Hopf 型蒙特卡洛参考：E3（必须报告 95% 置信区间，样本量 ≥ 10⁷，跑 CPU 分区）；
- 文献参考值：E4（多层 Picard，d=100 标准算例）。

### 基准方程组（选择标准：参考解可信 > 文献可比 > 有趣）

| 编号 | 方程 | 类型 | 参考解 | 维度范围 |
|------|------|------|--------|----------|
| E1 | Black–Scholes–Barenblatt：`u_t + ½σ²Σxᵢ²u_{xᵢxᵢ} = r(u − Σxᵢu_{xᵢ})` | 线性，金融 | 解析：`u = exp((r+σ²)(T−t))·‖x‖²` | 任意 d |
| E2 | LQ 最优控制的 HJB | 半线性，控制 | Riccati ODE 闭式解 | 任意 d |
| E3 | HJB（E–Han–Jentzen 标准例）：`u_t + Δu − λ‖∇u‖² = 0`，`g = ln((1+‖x‖²)/2)` | 完全非线性梯度项 | Cole–Hopf 变换 + MC（带 CI） | 任意 d，d=100 canonical |
| E4 | Allen–Cahn：`u_t + Δu + u − u³ = 0` | 半线性，反应扩散 | 文献值（MLP 方法，d=100，u(0,0) ≈ 0.0528） | d=100 为主 |
| E5 | 热方程（高斯初值） | 线性 | 解析 | sanity check 专用 |

> 主管备注：E1+E3 是最小可行故事（一线性一非线性、都有任意维参考）；E2 展示与控制理论（COCV 课程）的联系；E4 是社区标准算例便于与文献直接对表。若时间紧，砍 E2、E4 保 E1、E3。

## A3 实验矩阵

**主扫描（Experiment MAIN）**
方法 {M1a, M1b, TT-Tier2} × 方程 {E1, E2, E3, E4} × 维度 d ∈ {5, 10, 25, 50, 100} × 种子 {3}
≈ 每方法 60 runs；单 run 预计 10–60 min（单 GPU / 单 CPU 节点）。

**TT 表征扫描（Experiment RANK，Tier 1）**
方程 {E1, E2, E5} × d ∈ {5,10,25,50,100} × 目标精度 {1e-2, 1e-3, 1e-4} → 记录所需 TT 秩、内存、时间。纯 CPU。

**精度消融（Experiment PREC）**
在 E1、E3 × d ∈ {25, 100} 上：{fp32@5090, TF32@5090, fp64@A100} × 3 种子。回答 RQ4。

**扩展性小实验（Experiment SCALE，可选）**
E3 d=100 大 batch 训练，1 vs 2 GPU（torchrun / DDP），报告扩展效率。A100 分区 QoS 上限恰好是 2 卡——把上限跑满本身就是展示。

## A4 算力使用计划（这是竞赛评分的核心区，认真读）

### 关键约束（来自培训材料，务必以 `sinfo` / `sacctmgr` 实时输出为准）

| 分区 | QoS | 并发上限 | 适用任务 |
|------|-----|----------|----------|
| P107-A100（anode16-17，A100 80G ×16） | qos_p107-a100 | **MaxJobs=4，组内 cpu=16，gpu=2** | fp64 训练、大显存、NVLink 双卡 DDP |
| P107-RTX5090（anode01-15，5090 ×120） | qos_p107-rtx5090 | MaxJobs=4，组内 cpu=16，**gpu=4** | fp32/TF32 主力扫参 |
| CPU-6530（默认，14 节点常 idle） | 默认 | 宽松 | TT 全部实验、MC 参考解、数据/绘图 |

### 调度策略

1. **数组作业 + 节流**：主扫描用 `sbatch --array=1-60%4`（5090）/ `%2`（A100），每个 task 从 CSV 清单读取自己的 (方法, 方程, d, seed) 配置。CPU 侧 TT/MC 用独立数组作业分流，避免与 GPU 队列争 CPU 配额（GrpTRES cpu=16 是硬顶，GPU 作业每个只申请 `-c 4`）。
2. **断点续算（checkpoint/requeue）**：每个训练 run 定期落盘 checkpoint，脚本启动时自动检测续跑；配合 `scontrol requeue` 应对抢占。这直接命中比赛"高级功能：断点续算"的加分点。
3. **精度与硬件匹配**：fp64 → A100（19.5 TFLOPS FP64），fp32/TF32 → 5090（FP64 仅 3.4 TFLOPS，但 FP16/32 吞吐高且可 4 卡并发）。
4. **证据留存**：所有作业经 `sacct` 导出 CSV 入库；Grafana 面板截图（GPU 利用率、队列）；`sinfo`/`squeue` 快照。写进报告附录作为"真实使用平台"的证据。
5. **存储纪律**：代码与数据放 `/public/home/$USER/hdpde`（共享），`/tmp` 只作节点内临时目录（用完即清）；数据集用 HDF5，checkpoint 定期清理只留 best + last。
6. **环境**：`module load python3.12/3.12 cuda/13.0` + venv；pip 走 USTC 镜像源。登录节点只编辑/提交，**绝不跑训练**。

## A5 代码仓库设计（复用 learning-pdes 的成熟模式）

```text
hdpde/
├── pyproject.toml            # 依赖齐全（torch, numpy, scipy, h5py, teneva/torchtt, matplotlib, pyyaml, pytest）
├── CLAUDE.md                 # ← Prompt P0 的内容，AI 助手的常驻上下文
├── configs/
│   ├── defaults.yaml         # seed / device / precision / paths
│   ├── equations/            # e1_bsb.yaml ... e5_heat.yaml（d 作为可覆盖参数）
│   ├── methods/              # deep_bsde_orig.yaml, deep_bsde_shared.yaml, tt_solver.yaml, tt_cross.yaml
│   └── experiments/          # main.yaml, rank.yaml, prec.yaml, scale.yaml（声明 sweep 网格）
├── src/hdpde/
│   ├── equations/            # 基类 Equation：drift μ, diffusion σ, nonlinearity f, terminal g, exact/reference solution
│   ├── solvers/
│   │   ├── deep_bsde/        # 网络、时间离散、Trainer（AMP 开关、checkpoint/resume）
│   │   └── tt/               # tier1_cross.py, tier2_wrapper.py（封装外部求解器）
│   ├── reference/            # 解析解、Riccati 求解器、Cole–Hopf MC（带 CI）
│   ├── evaluation/           # 统一 metrics：rel_err@x0、CI、wall-clock、峰值显存、参数量/TT秩
│   └── utils/                # config、registry、seeding、device、io
├── scripts/
│   ├── run_one.py            # 跑单个 (method, equation, d, seed, precision) —— 数组作业的原子单元
│   ├── make_manifest.py      # 由 experiments/*.yaml 展开成 runs.csv 清单
│   ├── make_slurm.py         # 由清单生成带节流的 sbatch 数组脚本（分区/QoS 参数化）
│   ├── aggregate.py          # 扫 results/ 汇总成 summary.csv（含 sacct 信息合并）
│   └── plots.py              # 六张核心图一键生成
├── slurm/                    # 模板：gpu_array_5090.sbatch, gpu_array_a100.sbatch, cpu_array.sbatch, ddp_2gpu.sbatch
├── tests/                    # 方程解析解自检、求解器低维收敛性、config/registry
└── results/                  # <run_id>/{config.yaml, metrics.json, ckpt/, log}
```

> 主管备注：`run_one.py` 是全项目的原子接口——本地调试、数组作业、DDP 都调它。把它写干净，其他一切都是编排。

## A6 里程碑与时间线（6 周制，可压缩到 4 周）

| 周 | 里程碑 | 验收标准（硬性） |
|----|--------|------------------|
| W1 | 仓库骨架 + 方程库 + 参考解 | E1/E2/E5 解析解通过 pytest；E3 的 MC 参考在 d=100 给出带 CI 的值 |
| W2 | Deep BSDE 在低维验证收敛 | E1 d=1,5：相对误差 < 1%；E3 d=100 复现文献量级（~0.5% 内） |
| W3 | TT Tier1 完成 + Tier2 跑通 | RANK 扫描出图；Tier2 至少在 E1 上与解析解对上 |
| W4 | 集群主扫描（MAIN 全量） | 120+ runs 全部落盘，sacct 台账齐全，失败率 < 10% 且有 requeue 记录 |
| W5 | PREC + SCALE + 汇总分析 | 六张核心图定稿；RQ1–RQ4 各有一段可写的结论 |
| W6 | 报告 + 演示 + 打磨 | 技术报告 + 5 分钟 demo（现场提交一个作业并展示监控） |

## A7 风险登记与降级路径

| 风险 | 概率 | 应对 |
|------|------|------|
| R1 TT 求解器工程量失控 | 高 | 三层设计已内置：Tier1 兜底即成文；Tier2 只做封装不做修改 |
| R2 Deep BSDE 某些 (方程, d) 不收敛 | 中 | 用文献公开超参起步；梯度裁剪 + lr 衰减 + 3 种子；**失败模式如实报告，本身是 RQ5 的结果** |
| R3 A100 只有 2 卡并发导致排队 | 高 | fp32 主力全走 5090；A100 只跑 PREC 与 SCALE；数组节流 + checkpoint 续跑 |
| R4 MC 参考解自身误差污染结论 | 中 | E3 参考必须带 95% CI 并画进误差图；样本量做收敛性自检 |
| R5 时间不足 | 中 | 降级顺序：砍 RQ5 → 砍 E2/E4 → 砍 M1b 消融 → 保 E1+E3 × M1a × Tier1 的最小故事 |
| R6 集群环境装不上依赖 | 低 | 全部走 USTC pip 镜像；torchtt 装不上就用纯 numpy 的 teneva；无 sudo，一律 venv |

## A8 交付物清单（对齐评审视角）

1. **代码仓库**（git.ustc.edu.cn，含完整 Slurm 脚本与 README，可一键复现）；
2. **六张核心图**：① 误差 vs d（各方法）② 成本—精度 Pareto ③ TT 秩 vs d/精度 ④ fp32/fp64 精度—吞吐对比 ⑤ 1v2 GPU 扩展效率 ⑥ 多种子方差箱线图；
3. **实验台账**：runs.csv + sacct 导出 + Grafana 截图（平台使用证据）；
4. **技术报告**（10–15 页：方法、标度律结论、方法适用边界的讨论）；
5. **现场演示**：提交一个 d=100 作业 → squeue/Grafana 观察 → 展示已训练模型秒级推理 vs "网格法在 d=100 需要的存储量"数量级估算（讲维数灾难最直观的一页）。

---

# Part B · 结构化 Prompt 集（P0–P9）

## 使用说明

- 适用对象：Claude Code / 其他 AI 编程助手。**一个 Prompt 对应一个开发阶段**，按序执行，每阶段结束 `git commit` 后再进入下一个。
- P0 是常驻上下文：保存为仓库根目录的 `CLAUDE.md`（Claude Code 会自动读取），或在每次会话开头粘贴。
- 每个 Prompt 自带**验收标准**——验收不过不许进入下一阶段，让 AI 自己跑测试自证。
- 培训材料的忠告在此同样适用：**人决策、把方向、审结果；AI 跑腿。做好 git 备份，小改动可能捅大篓子。**

---

## P0 · 主上下文（保存为 CLAUDE.md）

```text
# 项目：hdpde —— 高维 PDE 的深度学习求解与张量列方法对比

## 你的角色
你是本项目的结对工程师。项目负责人是应用数学硕士生（PDE 数值分析 + 数据科学方向），
最终成果用于算力平台竞赛与课程研究。负责人负责科学决策，你负责实现与工程质量。

## 科学目标
在高维半线性抛物 PDE（d = 5 ~ 100）上对比：
- Deep BSDE 方法（E–Han–Jentzen 2018）：两个变体（每步子网络 / 共享网络）
- Tensor-Train 方法：TT-cross 表征研究（Tier1）+ 外部 TT 抛物求解器封装（Tier2）
核心问题：误差与成本随维数 d 的标度律；两类方法的适用边界；fp32 vs fp64 的影响。

## 基准方程（src/hdpde/equations/，每个必须带参考解）
- E1 Black–Scholes–Barenblatt：解析解 u = exp((r+σ²)(T−t))·‖x‖²
- E2 LQ 控制 HJB：Riccati ODE 闭式解
- E3 HJB（u_t + Δu − λ‖∇u‖² = 0, g = ln((1+‖x‖²)/2)）：Cole–Hopf + Monte Carlo 参考（带 95% CI）
- E4 Allen–Cahn（u_t + Δu + u − u³ = 0）：文献参考值 u(0,0) ≈ 0.0528（d=100, T=0.3）
- E5 热方程：解析解，仅作 sanity check

## 技术栈与硬性约束
- Python ≥ 3.10，PyTorch（GPU），numpy/scipy，h5py，pyyaml，pytest；TT 用 teneva（纯 numpy）优先，torchtt 备选
- 目标环境：Slurm 集群，无 sudo，module load python3.12 cuda；一切依赖必须 pip 可装（USTC 镜像）
- 所有随机性过 utils/seeding.py 统一控制；device 选择集中在 utils/device.py（cuda→cpu 回退）
- 精度是一等公民：所有 solver 接受 precision ∈ {fp32, tf32, fp64} 配置项
- 每个训练 run 必须支持 checkpoint 落盘与 --resume 自动续跑（集群断点续算）
- 原子接口：scripts/run_one.py --method M --equation E --dim D --seed S --precision P
  一切实验（本地/数组作业/DDP）都通过它

## 代码规范
- config 模式：dataclass + YAML 继承 + CLI 覆盖（参考 learning-pdes 项目的 src/config.py 风格）
- registry 模式：@register_equation / @register_method 装饰器
- 每个 run 输出到 results/<run_id>/：config.yaml（冻结的完整配置）、metrics.json、ckpt/、train.log
- metrics.json 至少含：rel_err_at_x0、reference_value、reference_ci、wall_clock_sec、
  peak_gpu_mem_mb、n_params_or_tt_ranks、precision、git_commit
- 不写死路径；所有路径来自 config；不用全局状态
- 每个模块配 pytest；数值断言给出容差与理由（注释里写清楚）

## 禁止事项
- 禁止在没有参考解的配置上报告"误差"
- 禁止静默捕获训练发散（NaN/Inf 必须显式记录 status=diverged 并正常退出）
- 禁止把大文件（>50MB checkpoint、数据集）提交进 git
```

---

## P1 · 仓库骨架与配置系统

```text
背景：CLAUDE.md 已就位。这是项目第一个开发任务。

任务：搭建 hdpde 仓库骨架。
1. pyproject.toml：项目元信息 + 全部依赖（含 dev extras: pytest, ruff）；确认 h5py、pyyaml 在列
2. 目录结构按 CLAUDE.md 的仓库设计创建，空模块给出带 docstring 的占位
3. 实现 src/hdpde/utils/：
   - config.py：dataclass 定义 + YAML 加载 + `inherits:` 键实现配置继承 + CLI 点路径覆盖（如 --method.lr 1e-3）
   - registry.py：register_equation / register_method 装饰器与 build_* 工厂
   - seeding.py（torch/numpy/random 三处统一）、device.py、io.py（results 目录管理，生成 run_id）
4. configs/defaults.yaml 与一个最小 configs/experiments/smoke.yaml
5. tests/test_config.py、tests/test_registry.py

验收标准：
- `pip install -e ".[dev]"` 在全新 venv 成功
- `pytest` 全绿
- `python -c "from hdpde.utils.config import load_config; print(load_config('configs/defaults.yaml'))"` 正常输出
- README 草稿含安装与目录说明
```

---

## P2 · 方程库与参考解（项目的地基，最高质量要求）

```text
背景：骨架已就位（P1）。本阶段实现全部基准方程与参考解。这是全项目正确性的地基，宁慢勿错。

任务：
1. src/hdpde/equations/base.py：抽象类 Equation，接口包括
   drift(t,x)、diffusion(t,x)、nonlinearity f(t,x,u,z)、terminal g(x)、
   x0（评估点）、T、以及 reference_solution(t,x) -> (value, ci or None)
2. 实现 E1（BSB）、E2（LQ-HJB）、E3（HJB-EHJ）、E4（Allen–Cahn）、E5（热方程），全部注册进 registry，
   维度 d 为构造参数
3. src/hdpde/reference/：
   - E2 的 Riccati ODE 求解（scipy.integrate.solve_ivp，容差 1e-10，任意 d）
   - E3 的 Cole–Hopf MC 参考：u(t,x) = −(1/λ)·ln E[exp(−λ g(x + √2 W_{T−t}))]，
     分批采样避免内存爆炸，返回值 + 95% CI；样本量可配置，默认 1e7
   - MC 自检：样本量减半时 CI 应约扩大 √2 倍（写成测试）
4. tests/test_equations.py：
   - E1/E5 解析解与已知特例交叉验证（如 d=1 手算值）
   - E2 Riccati 在 d=1 与手推闭式解对比，误差 < 1e-8
   - E3 MC 在 d=1 与数值积分（scipy.quad）对比，落在 CI 内
   - 所有方程在 d ∈ {1, 5, 50} 下形状/数值健全性检查

验收标准：pytest 全绿；给出一个表格（打印即可）：每个方程在 d=5 与 d=100 处 x0 的参考值与不确定度。
```

---

## P3 · Deep BSDE 求解器（M1a + M1b）

```text
背景：方程库已验收（P2）。实现两个 Deep BSDE 变体与统一 Trainer。

任务：
1. src/hdpde/solvers/deep_bsde/
   - discretization.py：Euler–Maruyama 前向路径模拟（batch 化，支持 fp32/tf32/fp64）
   - networks.py：
     M1a：u0 与 z0 为可训练参数 + 每个内部时间步一个子网络（结构：Linear-BN-ReLU ×2，宽度 d+10，按原文）
     M1b：单个共享网络 φ(t, x)（sin 或 tanh 激活），u0 仍为可训练标量
   - trainer.py：损失 = E|g(X_T) − Y_T|²；Adam；lr schedule 与梯度裁剪可配置；
     AMP 开关（tf32 用 torch.backends 配置，fp64 全程 double）；
     每 k 步记录 {loss, u0_estimate, rel_err_vs_reference, lr, wall_clock}；
     checkpoint 落盘（last + best）与 --resume 自动检测续跑
2. scripts/run_one.py 打通：--method deep_bsde_orig --equation e3_hjb --dim 100 --seed 0 --precision fp32
3. configs/methods/deep_bsde_orig.yaml 与 deep_bsde_shared.yaml：
   默认超参对齐原文（时间步数 N=20~40，batch 64~256，迭代 2000~10000，lr 1e-2 阶梯衰减）
4. tests/test_deep_bsde.py：E5 热方程 d=2 上 500 步内 rel_err < 5%（快速冒烟测试，CPU 可跑）

验收标准（在有 GPU 的环境跑，无 GPU 则报告 CPU 缩水版并注明）：
- E1 d=5：rel_err < 1%
- E3 d=100：u(0,x0) 复现文献量级（相对 MC 参考误差 < 1%，且落在或接近其 CI）
- 中断-续跑测试：kill 后 --resume 能从 checkpoint 继续且 loss 曲线连续
- 训练日志与 metrics.json 字段齐全（对照 CLAUDE.md 清单）
```

---

## P4 · TT 基线：Tier1 表征研究 + Tier2 求解器封装

```text
背景：Deep BSDE 已验收（P3）。本阶段建立张量列对照组。分两个独立子任务，Tier1 优先。

任务 Tier1（TT-cross 表征研究，纯 CPU）：
1. src/hdpde/solvers/tt/tier1_cross.py：
   - 用 teneva 的 TT-cross 逼近 equation.reference_solution 在盒域 [x0−L, x0+L]^d 上的函数
   - 输入：方程、d、目标相对精度 eps；输出：达到 eps 所需 TT 秩谱、元素数压缩比、构建时间
   - 精度用独立随机测试点集上的相对 L2 误差衡量（不少于 1e4 点）
2. 注册为 method=tt_cross，走 run_one.py 统一接口，metrics.json 记录 max_rank / ranks / n_params_tt
3. tests：E5 d=3 上低秩解应在 rank ≤ 5 内达到 1e-3（写成断言，容差注明理由）

任务 Tier2（外部 TT 抛物求解器封装）：
1. 调研并接入一个公开的 TT 高维抛物 PDE 求解器实现
   （优先：Richter–Sallandt–Nüsken, ICML 2021 的官方代码）。以 git submodule 或 vendored 目录引入，
   保留上游 LICENSE 与出处注释
2. src/hdpde/solvers/tt/tier2_wrapper.py：把我们的 Equation 对象翻译成上游求解器的问题定义，
   跑完把结果转回统一 metrics.json；method=tt_solver
3. 上游代码不可用/接口对不上时：如实报告阻塞点与两个备选方案，停下来等负责人决策，不要自行硬改上游算法

验收标准：
- Tier1：E1 在 d ∈ {5,10,25} 的秩-精度表打印成 markdown 表格
- Tier2：至少 E1 d=10 上 rel_err < 1%（对解析解）
- 两者均可通过 run_one.py 调用且输出格式与 Deep BSDE 完全一致
```

---

## P5 · 实验编排：清单、Slurm 数组作业、断点续跑

```text
背景：三条方法线均已打通（P3、P4）。本阶段建设集群规模化实验的编排层。

集群事实（写死进模板的参数化默认值，但全部可配置）：
- 分区 P107-RTX5090：QoS qos_p107-rtx5090，组内并发 gpu=4 / cpu=16 → 数组节流 %4，每作业 --gres=gpu:1 -c 4
- 分区 P107-A100：QoS qos_p107-a100，组内并发 gpu=2 / cpu=16 → 数组节流 %2，每作业 --gres=gpu:1 -c 4
- CPU 分区 CPU-6530：TT 与 MC 参考解在此跑，每作业 -c 16 起步
- 模块：module load python3.12/3.12 cuda/13.0；家目录 /public/home/$USER（共享），/tmp 节点独立
- 登录节点不跑计算

任务：
1. scripts/make_manifest.py：读 configs/experiments/*.yaml 的 sweep 声明
   （axes: method/equation/dim/seed/precision + 排除规则），展开为 runs.csv
   （每行 = 一个 run_one.py 调用 + 目标分区 + 预计时长），支持 --dry-run 打印统计
2. scripts/make_slurm.py：由 runs.csv 按分区分组生成 sbatch 脚本：
   - 数组作业，$SLURM_ARRAY_TASK_ID 索引 CSV 行
   - 节流参数、--time、-J、-o %A_%a.out 全参数化
   - 作业体：module load → source venv → python scripts/run_one.py ... --resume
   - TIMEOUT 前自动落 checkpoint（trap SIGTERM）+ 可选 scontrol requeue 自身
3. slurm/ 下生成三个模板实例：gpu_array_5090.sbatch、gpu_array_a100.sbatch、cpu_array.sbatch
4. scripts/collect_sacct.py：按作业名前缀拉取 sacct（JobID,State,Elapsed,MaxRSS,ExitCode…）
   合并进 runs.csv 形成台账 ledger.csv
5. configs/experiments/：main.yaml（A3 的主扫描）、rank.yaml、prec.yaml 三个真实实验定义

验收标准：
- make_manifest --dry-run 对 main.yaml 输出正确的 run 总数与分区分布
- 生成的 sbatch 脚本通过 shellcheck（或人工 review 清单）
- 本地模拟：SLURM_ARRAY_TASK_ID=3 bash -x slurm/cpu_array.sbatch 能正确取到第 3 行并启动 run_one.py
- 文档：docs/cluster.md 写清"从零到提交主扫描"的完整命令序列
```

---

## P6 · 汇总分析与六张核心图

```text
背景：主扫描已在集群完成（或部分完成），results/ 下有 ≥ 50 个 run。

任务：
1. scripts/aggregate.py：递归扫 results/*/metrics.json + 合并 ledger.csv → summary.csv
   （一行一 run，列含全部配置轴与全部指标；缺失/发散 run 标记 status）
2. scripts/plots.py 生成六张图（matplotlib，出版级：标签/图例/字号统一，同时存 pdf+png）：
   F1 误差 vs d（log-log，按方法分色，按方程分面板；E3 画出 MC 参考 CI 带）
   F2 成本—精度 Pareto（x=wall-clock，y=rel_err，log-log，点形=方法，点色=d）
   F3 TT 秩 vs d，按目标精度分线（Tier1 数据）
   F4 精度消融：fp32/tf32/fp64 的 rel_err 柱状 + 吞吐（it/s）双轴，按分区标注
   F5 多种子箱线图：每 (方程, d) 一组，展示 Deep BSDE 方差
   F6 扩展性：1 vs 2 GPU 吞吐与效率（若 SCALE 实验有数据，否则跳过并注明）
3. 每张图配一段 3–5 句的"图注结论"写进 docs/findings.md（描述模式，不过度解读）
4. 稳健性：任何一张图在数据不全时应降级绘制并警告，而不是崩溃

验收标准：单命令 `python scripts/plots.py --summary results/summary.csv --out figs/` 产出全部图；
figs/ 与 findings.md 提交 git；负责人抽查 3 个 run 的原始 metrics 与 summary 行一致。
```

---

## P7 · 精度—硬件研究（RQ4 专项）

```text
背景：主结果已出（P6）。本阶段回答"fp32 够不够、A100 何时必要"。

任务：
1. 在 trainer 中确认三种模式的实现正确性：
   - fp32：默认；tf32：允许 matmul/conv 用 TF32（torch.backends.cuda.matmul.allow_tf32 等）；
   - fp64：模型、数据、路径模拟全程 double（注意 E3 的 exp/log 数值稳定性）
2. 设计对照：E1、E3 × d ∈ {25, 100} × 3 种子 × {fp32@5090, tf32@5090, fp64@A100}
   附加一组 fp64@5090（预期极慢，跑 1 个短 run 仅测吞吐，用于量化 6 倍 FP64 差距）
3. 指标：最终 rel_err、收敛所需迭代数、it/s、峰值显存、（可得则）能耗估算
4. 分析脚本输出一个决策表：每个 (方程, d) 给出"最经济精度选择"及理由
5. 更新 F4 图与 findings.md

验收标准：决策表能回答"本项目哪些实验其实不需要 A100"；
fp64@5090 与 fp64@A100 的吞吐比与硬件规格（3.4 vs 19.5 TFLOPS）数量级一致（否则解释原因）。
```

---

## P8 · 技术报告与竞赛材料

```text
背景：全部实验与图表定稿。本阶段产出报告初稿与竞赛演示材料。负责人提供最终科学判断，你负责成文结构与一致性。

任务：
1. docs/report/ 下用 markdown（或 LaTeX，负责人指定）搭报告骨架并填充初稿：
   摘要 / 1 引言（维数灾难，两条打破路线）/ 2 方法（BSDE 表示 + Deep BSDE 两变体 + TT 两层）/
   3 基准方程与参考解（含 CI 方法学）/ 4 实验设置（集群、QoS、调度策略——把 A4 写进去）/
   5 结果（六图 + findings）/ 6 讨论（适用边界：低秩结构 vs 非线性强度；精度—硬件匹配）/
   7 结论与展望 / 附录（超参表、台账统计、复现命令）
2. 所有数字必须从 summary.csv 引用（写一个 docs/numbers.py 把关键数字生成为 include 片段，防手抄错）
3. 演示脚本 docs/demo.md：5 分钟流程——现场 sbatch 提交 E3 d=100 → squeue/Grafana 展示 →
   加载已训练 checkpoint 秒级推理 → 展示"d=100 网格法需要 10^200 量级网格点"的对比页
4. README 终稿：安装、复现主扫描的完整命令、结果目录说明、引用文献列表
   （E–Han–Jentzen 2018；Richter–Sallandt–Nüsken 2021；相关综述）

验收标准：报告初稿全文无 TODO 占位；每个图表编号被正文引用；复现命令在干净 clone 上走通到 smoke 级别。
```

---

## P9 · 调试请求模板（贯穿全程使用）

```text
向 AI 求助排障时，按此模板提供信息（源自培训"提问的智慧"，同样适用于 AI）：

【环境】分区/节点、module list、python/torch/cuda 版本、git commit
【命令】完整的复现命令（run_one.py 参数 或 sbatch 脚本名 + array id）
【现象】期望 vs 实际；完整报错栈；相关日志片段（.out/.err 的关键 50 行，不要只给最后一行）
【已排查】已尝试的 2–3 件事及结果
【约束】不许改的部分（如上游 TT 代码、已冻结的配置）

要求 AI：先给出诊断假设排序，再给最小侵入的修复方案；涉及数值正确性的修改必须附带新的回归测试。
```

---

## 附 · 主管的三条底线（打印贴墙）

1. **没有参考解，就没有"误差"**——任何图上的每一个点都必须能回答"和什么比"。
2. **失败的 run 也是数据**——发散、超时、排队都如实入台账；隐藏失败等于放弃 RQ5 且违反科学诚信。
3. **每周五 git tag**——`w1-equations`、`w2-bsde-validated`……任何时候能回滚到上一个可用状态。
