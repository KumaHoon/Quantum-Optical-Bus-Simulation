# 量子光学总线 - 标定看板

**Languages:** [English](../README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [中文](README.zh.md)

这是一个<strong><i lang="en">One Waveguide (Hardware), Infinite States (Software)</i></strong>的混合量子-经典仿真项目。

---

## 实时演示

`calibration_demo.gif` 展示了 0–200 mW 的泵浦功率扫描（固有压缩）和 0–2 dB 的损耗扫描（观测压缩下降）。

<p align="center"><img src="../assets/calibration_demo.gif" width="950" alt="Live demo animation" /></p>

> **Figure 1: 实时校准演示.**
> <i lang="en">Intrinsic Squeezing (pre-loss)</i> 随输入功率固定变化，
> <i lang="en">Observed Squeezing (post-loss)</i> 随损耗降低。

---

## 架构

```mermaid
flowchart LR
    UI["<i lang=\"en\">calibration_app.py</i><br/>输入: P, θ, loss_dB/cm, length"]
    HW["<i lang=\"en\">hardware.py</i><br/>模式剖面 + n<sub>eff</sub>"]
    I["<i lang=\"en\">interface.py</i><br/>P -> r"]
    U["<i lang=\"en\">units.py</i><br/>total_loss_dB -> eta_loss"]
    Q["<i lang=\"en\">quantum.py</i><br/><i lang=\"en\">run_single_mode</i>"]
    MM["<i lang=\"en\">multimode.py</i><br/><i lang=\"en\">run_multimode</i>"]
    TP["<i lang=\"en\">tdm_topology.py</i><br/><i lang=\"en\">simulate_topology</i>"]
    ES["<i lang=\"en\">estimation.py</i><br/><i lang=\"en\">fit_eta_and_loss</i>"]
    CT["<i lang=\"en\">control.py</i><br/><i lang=\"en\">apply_feedback_with_latency</i>"]
    O["<i lang=\"en\">calibration_app.py</i><br/>可视化 + 指标"]

    UI --> I
    UI --> U
    UI --> HW
    UI --> MM
    UI --> TP
    UI --> ES
    UI --> CT
    I --> Q
    U --> Q
    Q --> O
    MM --> O
    TP --> O
    ES --> O
    CT --> O
    HW --> O
```

## 场景画廊

| 场景 | 图片 |
|---|---|
| 1. Vacuum Baseline (P = 0 mW) | ![Vacuum Baseline](../assets/dashboard_vacuum.png) |
| 2. Squeezed State (P = 200 mW) | ![Calibration + Squeezing](../assets/dashboard_calibration.png) |
| 3. Decoherence (Pure vs Lossy) | ![Decoherence Comparison](../assets/dashboard_decoherence.png) |

<details>
<summary>场景 GIF</summary>
<p align="center"><img src="../assets/scenario_gallery.gif" width="950" alt="Scenario gallery animation" /></p>
</details>

## Advanced 画廊

| 场景 | 图片 |
|---|---|
| 4. 多模/时间窗 | ![Multi-mode Dashboard](../assets/dashboard_multimode.png) |
| 5. 拓扑 | ![Topology Dashboard](../assets/dashboard_topology.png) |
| 6. 数字孪生 + 控制 | ![Digital Twin Dashboard](../assets/dashboard_digital_twin.png) |

<details>
<summary>Advanced GIF</summary>
<p align="center"><img src="../assets/advanced_gallery.gif" width="950" alt="Advanced gallery animation" /></p>
</details>

## Evidence 画廊

<details>
<summary>Evidence GIF</summary>
<p align="center"><img src="../assets/advanced_evidence.gif" width="950" alt="Advanced evidence animation" /></p>
</details>

## 快速开始

```bash
pip install -e .
streamlit run src/quantum_optical_bus/calibration_app.py
```

打开 **http://localhost:8501** 后，通过侧边栏调整参数。

---

### 额外命令

| Task | Command |
|------|---------|
| 生成场景图片 | `python scripts/generate_dashboard_gallery.py` |
| 生成 Advanced 场景图片 | `python scripts/generate_advanced_dashboard_gallery.py` |
| 生成 Scenario GIF | `python scripts/generate_scenario_gallery_gif.py` |
| 生成 Advanced GIF | `python scripts/generate_advanced_gallery_gif.py` |
| 生成 Evidence GIF | `python scripts/generate_advanced_evidence_gif.py` |
| 生成演示 GIF | `python scripts/generate_calibration_demo.py` |
