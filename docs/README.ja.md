# 量子光学バス - キャリブレーションダッシュボード

**Languages:** [English](../README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [中文](README.zh.md)

<i lang="en">One Waveguide (Hardware), Infinite States (Software)</i> を用いた
ハイブリッド量子-古典シミュレーションです。クラシカルなポンプ電力を連続変数量子状態へ対応付けます。

---

## ライブデモ

`calibration_demo.gif` は 0–200 mW のポンプ増加（intrinsic）と 0–2 dB の損失増加（observed の減少）を示します。

<p align="center"><img src="../assets/calibration_demo.gif" width="950" alt="Live demo animation" /></p>

> **Figure 1: リアルタイムキャリブレーションシミュレーション.**  
> <i lang="en">Intrinsic Squeezing (pre-loss)</i> は所与のポンプ電力で一定、<i lang="en">Observed Squeezing (post-loss)</i> は損失で低下します。

---

## アーキテクチャ

```mermaid
flowchart LR
    UI["<i lang=\"en\">calibration_app.py</i><br/>入力: P, θ, loss_dB/cm, length"]
    HW["<i lang=\"en\">hardware.py</i><br/>モード分布 + n<sub>eff</sub>"]
    I["<i lang=\"en\">interface.py</i><br/>P -> r"]
    U["<i lang=\"en\">units.py</i><br/>total_loss_dB -> eta_loss"]
    Q["<i lang=\"en\">quantum.py</i><br/><i lang=\"en\">run_single_mode</i>"]
    MM["<i lang=\"en\">multimode.py</i><br/><i lang=\"en\">run_multimode</i>"]
    TP["<i lang=\"en\">tdm_topology.py</i><br/><i lang=\"en\">simulate_topology</i>"]
    ES["<i lang=\"en\">estimation.py</i><br/><i lang=\"en\">fit_eta_and_loss</i>"]
    CT["<i lang=\"en\">control.py</i><br/><i lang=\"en\">apply_feedback_with_latency</i>"]
    O["<i lang=\"en\">calibration_app.py</i><br/>表示 + 指標"]

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

| 層 | ファイル | 責務 |
|---|---|---|
| <i lang="en">Hardware</i> | `src/quantum_optical_bus/hardware.py` | 観測データ表示用の基盤情報。現行では直接 <i lang="en">r</i> を決定しない。 |
| <i lang="en">Interface</i> | `src/quantum_optical_bus/interface.py` | ポンプ電力 $P$ を $r=\eta\sqrt{P}$ へ写像。 |
| <i lang="en">Units</i> | `src/quantum_optical_bus/units.py` | 損失 dB を透過率 $\eta_{\text{loss}}$ へ変換。 |
| <i lang="en">Quantum</i> | `src/quantum_optical_bus/quantum.py` | `Sgate`, `Rgate`, `LossChannel` を用いた Wigner と評価値出力。 |
| <i lang="en">Multimode</i> | `src/quantum_optical_bus/multimode.py` | 時間ビン独立シミュレーション。 |

---

## シナリオギャラリー

| シナリオ | 画像 |
|---|---|
| 1. Vacuum Baseline (P = 0 mW) | ![Vacuum Baseline](../assets/dashboard_vacuum.png) |
| 2. Squeezed State (P = 200 mW) | ![Calibration + Squeezing](../assets/dashboard_calibration.png) |
| 3. Decoherence (Pure vs Lossy) | ![Decoherence Comparison](../assets/dashboard_decoherence.png) |

<details>
<summary>シナリオGIF</summary>

<p align="center"><img src="../assets/scenario_gallery.gif" width="950" alt="Scenario gallery animation" /></p>
</details>

## Advanced ギャラリー

| シナリオ | 画像 |
|---|---|
| 4. マルチモード / タイムビン | ![Multi-mode Dashboard](../assets/dashboard_multimode.png) |
| 5. トポロジー | ![Topology Dashboard](../assets/dashboard_topology.png) |
| 6. デジタルツイン + 制御 | ![Digital Twin Dashboard](../assets/dashboard_digital_twin.png) |

<details>
<summary>Advanced GIF</summary>

<p align="center"><img src="../assets/advanced_gallery.gif" width="950" alt="Advanced gallery animation" /></p>
</details>

## Evidence Gallery

<details>
<summary>Evidence GIF</summary>

<p align="center"><img src="../assets/advanced_evidence.gif" width="950" alt="Advanced evidence animation" /></p>
</details>

## クイックスタート

```bash
pip install -e .
streamlit run src/quantum_optical_bus/calibration_app.py
```

ブラウザで **http://localhost:8501** を開き、サイドバーでパラメータを変更してください。

---

### 追加コマンド

| Task | Command |
|------|---------|
| ギャラリー画像生成 | `python scripts/generate_dashboard_gallery.py` |
| Advanced ギャラリー画像生成 | `python scripts/generate_advanced_dashboard_gallery.py` |
| Scenario GIF生成 | `python scripts/generate_scenario_gallery_gif.py` |
| Advanced GIF生成 | `python scripts/generate_advanced_gallery_gif.py` |
| Evidence GIF生成 | `python scripts/generate_advanced_evidence_gif.py` |
| デモGIF生成 | `python scripts/generate_calibration_demo.py` |
