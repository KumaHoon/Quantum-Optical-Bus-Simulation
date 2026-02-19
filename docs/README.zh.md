# 量子光学总线 - 校准仪表板

[English](../README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | 中文

这是一个混合量子-经典模拟项目，核心口号是 **One Waveguide (Hardware), Infinite States (Software)**。  
它通过从经典泵浦功率映射到连续变量(CV)压缩参数 $r=\eta\sqrt{P}$ 来展示“校准 + 可视化”流程，并在仿真中评估损耗与测量结果。

---

## 实时演示

`calibration_demo.gif` 展示了 0–200 mW 泵浦扫描（intrinsic）与 0–2 dB 损耗扫描（observed）对应的 Wigner 形状变化。

![Live demo animation](../assets/calibration_demo.gif)

> **Figure 1: 实时校准演示。**  
> Intrinsic Squeezing (pre-loss) 由泵浦决定并较为稳定，  
> Observed Squeezing (post-loss) 随损耗增加而降低。

---

## 架构

```mermaid
flowchart TD
  User([Researcher / Operator]) --> App["Streamlit UI<br/>calibration_app.py<br/>orchestrator"]

  subgraph HOT["Hot path: input → mapping/units → simulation → outputs"]
    Mapping["interface.py<br/>P → r mapping (r=η√P)"]
    UnitMap["units.py<br/>loss(dB) ↔ transmissivity T"]
    Quantum["quantum.py<br/>Sgate / Rgate / LossChannel"]
    Output["Wigner + covariance + metrics"]
  end

  App -->|sliders: P, loss, θ, topology| Mapping --> Quantum
  App -->|loss in dB| UnitMap --> Quantum
  Quantum --> Output --> App

  subgraph OPT["Optional model/engine path"]
    HW["hardware.py<br/>Meep optional / analytical mock"]
    SF["Strawberry Fields<br/>(Gaussian backend)"]
    Meep["Meep<br/>mode profile / Aeff / n_eff"]
  end

  App -->|hardware params| HW
  HW -.->|optional profile| App
  Quantum -.-> SF
  HW -.-> Meep
  HW -.-> Ctxt["mode constraints to<br/>interface/units"]
  Ctxt --> Mapping
  Ctxt --> UnitMap

  subgraph FB["Feedback path"]
    Est["estimation.py<br/>fit η and loss"]
    Ctrl["control.py<br/>phase drift + latency feedback"]
  end

  App -->|measured variance / squeezing curves| Est
  Est -->|eta_hat, loss_hat| Ctrl
  Ctrl -->|residual / correction| App

  App --> Multi["multimode.py<br/>mode-wise time-bin pipeline"]
  App --> Top["tdm_topology.py<br/>topology + couplings"]
  Multi -->|mode metrics| App
  Top -->|correlations| App
```

`calibration_app.py` 作为总控器，读取侧栏参数，调用计算模块并更新 Wigner 图、covariance 指标和控制/拟合诊断。

### 责任表

| 层 | 文件 | 职责 |
|---|---|---|
| 硬件 | `src/quantum_optical_bus/hardware.py` | 支持 Meep 的模式估计；环境不支持时回退到分析 mock |
| 映射 | `src/quantum_optical_bus/interface.py` | 定义输入功率到 $r=\eta\sqrt{P}$ 的映射 |
| 单位换算 | `src/quantum_optical_bus/units.py` | dB 损耗与透过率转换，统一可视化标度 |
| 量子引擎 | `src/quantum_optical_bus/quantum.py` | `run_single_mode` 使用 Sgate/Rgate/LossChannel 并计算关键指标 |
| 多模扩展 | `src/quantum_optical_bus/multimode.py` | 每个模式独立处理 time-bin 结构 |
| 拓扑 | `src/quantum_optical_bus/tdm_topology.py` | 从配置构建 BS 连接序列并应用模式损耗 |
| 拟合 | `src/quantum_optical_bus/estimation.py` | `fit_eta_and_loss` 估计 η 与损耗 |
| 控制 | `src/quantum_optical_bus/control.py` | 含延迟的简化相位漂移反馈 |
| 编排 UI | `src/quantum_optical_bus/calibration_app.py` | 连接各模块并渲染场景流程 |

### 路线图说明（基于源码）

- `hardware.py` 保留硬件建模路径，但当前尚未演进为完整的在线标定提取器。
- `interface.py` 使用固定耦合参数的现象学映射，而非重叠积分得到的实测标定。
- `tdm_topology.py` 采用静态耦合序列的 MVP 实现，不包含完整时序抖动与硬件下发层。

更多模块交互说明见 [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)。

---

## 硬件在环扩展

![Hardware-in-the-loop expansion flow](figures/hil_expansion.png)

当前路线图包含：

- 光学路径（laser/OPA/loop/homodyne）
- 控制路径（ADC/FPGA/DAC/EOM driver）
- 世界模型路径（`estimation.py` → 控制系数更新 → `hdl` 部署）

---

## 场景画廊

| 场景 | 图像 |
|---|---|
| **1. Vacuum Baseline (P = 0 mW)** | ![Vacuum Baseline](../assets/dashboard_vacuum.png) |
| **2. Squeezed State (P = 200 mW)** | ![Calibration + Squeezing](../assets/dashboard_calibration.png) |
| **3. Decoherence (Pure vs Lossy)** | ![Decoherence Comparison](../assets/dashboard_decoherence.png) |

### 场景 GIF

![Scenario gallery animation](../assets/scenario_gallery.gif)

---

## 高级画廊

| 场景 | 图像 |
|---|---|
| **4. Multi-mode / Time-bin Simulator** | ![Multi-mode Dashboard](../assets/dashboard_multimode.png) |
| **5. Topology Simulator** | ![Topology Dashboard](../assets/dashboard_topology.png) |
| **6. Digital Twin + Control** | ![Digital Twin Dashboard](../assets/dashboard_digital_twin.png) |

### 高级 GIF

![Advanced gallery animation](../assets/advanced_gallery.gif)

---

## Evidence 画廊

### Evidence GIF

![Advanced evidence animation](../assets/advanced_evidence.gif)

---

## 快速开始

```bash
# 安装
pip install -e .

# 启动仪表板
streamlit run src/quantum_optical_bus/calibration_app.py
```

打开 **http://localhost:8501**，使用侧边栏调节参数。

### Docker 快速开始

在 Python 3.10 下已验证与 Strawberry Fields 的兼容性。

```bash
docker build .
docker compose up --build
```

### 附加命令

| Task | Command |
|---|---|
| 生成场景图片 | `python scripts/generate_dashboard_gallery.py` |
| 生成高级场景图片 | `python scripts/generate_advanced_dashboard_gallery.py` |
| 生成场景 GIF | `python scripts/generate_scenario_gallery_gif.py` |
| 生成高级 GIF | `python scripts/generate_advanced_gallery_gif.py` |
| 生成 Evidence GIF | `python scripts/generate_advanced_evidence_gif.py` |
| 生成演示 GIF | `python scripts/generate_calibration_demo.py` |

### 任务命令

```bash
make test
make lint
make app
```

---

## 模型定义与假设

### 压缩参数与控制量

```math
r = \eta\sqrt{P}
```

### 损耗模型

(代码变量为 `loss_dB`，数式写作 $\mathrm{loss}_{\mathrm{dB}}$ 以避免 underscore 解析报错)

```math
T = 10^{-\frac{\mathrm{loss}_{\mathrm{dB}}}{10}}
```

```math
\hat{a}_{\mathrm{out}} = \sqrt{T}\,\hat{a}_{\mathrm{in}} + \sqrt{1-T}\,\hat{a}_{\mathrm{vac}}
```

---

## 测试与 CI

```bash
pip install -e ".[test]"
python -m pytest -q
```

---

## 项目结构

```text
.
├── .github/workflows/ci.yml
├── src/
│   └── quantum_optical_bus/
│       ├── calibration_app.py
│       ├── quantum.py
│       ├── multimode.py
│       ├── tdm_topology.py
│       ├── estimation.py
│       ├── control.py
│       ├── hardware.py
│       ├── interface.py
│       ├── units.py
│       └── compat.py
├── tests/
├── scripts/
└── assets/
```
