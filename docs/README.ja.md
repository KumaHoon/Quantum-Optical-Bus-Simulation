# 量子光学バス - 校正ダッシュボード

[English](../README.md) | 日本語 | [한국어](README.ko.md) | [中文](README.zh.md)

One Waveguide (Hardware), Infinite States (Software)を用いた、光量子計算とCV量子状態を結びつけるハイブリッドなシミュレーションです。  
ポンプ出力を固定した量子状態制御変数 $r=\eta\sqrt{P}$ と損失後観測値にマッピングし、キャリブレーションとデジタルツインを可視化します。

---

## ライブデモ

`calibration_demo.gif` は、ポンプ 0–200 mW の掃引（intrinsic）と、損失 0–2 dB の掃引（observed）による Wigner 形状変化を示します。

![Live demo animation](../assets/calibration_demo.gif)

> **Figure 1: リアルタイム校正シミュレーション。**  
> Intrinsic Squeezing (pre-loss) はポンプで決まり一定値になり、  
> Observed Squeezing (post-loss) は損失で低下します。

---

## アーキテクチャ

```mermaid
flowchart LR
  User([User])

  subgraph UI["UI / App (Streamlit)"]
    App["calibration_app.py<br/>orchestrator + rendering"]
  end

  subgraph Lib["quantum_optical_bus (Python package)"]
    Interface["interface.py<br/>P -> r mapping"]
    Units["units.py<br/>loss dB <-> T, scaling"]
    Quantum["quantum.py<br/>single-mode Gaussian ops"]
    Multi["multimode.py<br/>independent multi-mode/time-bin"]
    Topology["tdm_topology.py<br/>BS couplings by config"]
    Est["estimation.py<br/>fit eta & loss"]
    Ctrl["control.py<br/>phase drift + latency feedback"]
    HW["hardware.py<br/>optional Meep / analytic mock"]
  end

  subgraph Ext["External deps (optional)"]
    SF["Strawberry Fields<br/>(Gaussian backend)"]
    Meep["Meep (optional)<br/>eigenmode estimate"]
  end

  User -->|sliders: P, loss, theta, topology| App

  App -->|mode view / params| HW
  App -->|r=eta*sqrt(P)| Interface
  App -->|loss in dB| Units

  Interface --> Quantum
  Units --> Quantum
  Quantum -->|Wigner, cov, metrics| App

  App --> Multi
  App --> Topology
  Multi -->|per-mode metrics| App
  Topology -->|correlations| App

  App --> Est
  App --> Ctrl
  Est -->|eta_hat, loss_hat| App
  Ctrl -->|residual/error metrics| App

  Quantum -.-> SF
  Multi -.-> SF
  Topology -.-> SF
  HW -.-> Meep
```

`calibration_app.py` はUIの入力を集約し、計算モジュールを呼び出して Wigner 表示、二次モーメント、制御・適合診断を更新する調停役です。

### 責任表

| 層 | ファイル | 責務 |
|---|---|---|
| ハードウェア | `src/quantum_optical_bus/hardware.py` | Meepを使える場合は固有モードを推定し、未対応環境では解析モデルへフォールバック |
| 変換 | `src/quantum_optical_bus/interface.py` | 入力ポンプと圧縮係数の対応関係 $r=\eta\sqrt{P}$ を定義 |
| 単位 | `src/quantum_optical_bus/units.py` | dB損失と透過率の換算、および可視化で用いるスケーリングを提供 |
| 量子エンジン | `src/quantum_optical_bus/quantum.py` | `run_single_mode` が Sgate/Rgate/LossChannel を構築し、観測量と covariance 指標を返す |
| 多モード拡張 | `src/quantum_optical_bus/multimode.py` | モードごとに独立な time-bin ガウス回路を評価 |
| トポロジ | `src/quantum_optical_bus/tdm_topology.py` | 設定トポロジーに基づくBS結合と経路損失を再現 |
| 推定 | `src/quantum_optical_bus/estimation.py` | `fit_eta_and_loss` による η および損失の同定 |
| 制御 | `src/quantum_optical_bus/control.py` | 位相ドリフトと遅延を含む簡略制御ループを提供 |
| UI | `src/quantum_optical_bus/calibration_app.py` | サイドバー入力を受け取り、ワークフローを表示 |

### ロードマップ（実装ベース）

- `hardware.py` は将来のキャリブレーション抽出を想定した構造ですが、現時点では推定パイプラインを完全代替していません。
- `interface.py` は重なり積分の実測値ではなく、固定係数での現象論的マッピングを採用しています。
- `tdm_topology.py` は現状静的な結合列を使う MVP で、完全な時間同期/ハードウェア制御統合までは含みません。

詳細は [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) を参照してください。

---

## ハードウェアインザループ拡張

![Hardware-in-the-loop expansion flow](figures/hil_expansion.png)

現在のロードマップは以下です。

- 光学経路（laser/OPA/loop/homodyne）
- 制御経路（ADC/FPGA/DAC/EOM driver）
- 世界モデル経路（`estimation.py` → 制御係数更新 → `hdl` 展開）

---

## シナリオギャラリー

| シナリオ | 画像 |
|---|---|
| **1. Vacuum Baseline (P = 0 mW)** | ![Vacuum Baseline](../assets/dashboard_vacuum.png) |
| **2. Squeezed State (P = 200 mW)** | ![Calibration + Squeezing](../assets/dashboard_calibration.png) |
| **3. Decoherence (Pure vs Lossy)** | ![Decoherence Comparison](../assets/dashboard_decoherence.png) |

### シナリオ GIF

![Scenario gallery animation](../assets/scenario_gallery.gif)

---

## 高度シナリオギャラリー

| シナリオ | 画像 |
|---|---|
| **4. Multi-mode / Time-bin Simulator** | ![Multi-mode Dashboard](../assets/dashboard_multimode.png) |
| **5. Topology Simulator** | ![Topology Dashboard](../assets/dashboard_topology.png) |
| **6. Digital Twin + Control** | ![Digital Twin Dashboard](../assets/dashboard_digital_twin.png) |

### 高度シナリオ GIF

![Advanced gallery animation](../assets/advanced_gallery.gif)

---

## Evidence ギャラリー

### Evidence GIF

![Advanced evidence animation](../assets/advanced_evidence.gif)

---

## クイックスタート

```bash
# Install
pip install -e .

# Dashboard 起動
streamlit run src/quantum_optical_bus/calibration_app.py
```

ブラウザで **http://localhost:8501** を開き、サイドバーで設定を調整します。

### Docker クイックスタート

Strawberry Fields と Python 3.10 の互換実行環境で動作確認済みです。

```bash
docker build .
docker compose up --build
```

### 追加コマンド

| Task | Command |
|---|---|
| Dashboard 画像生成 | `python scripts/generate_dashboard_gallery.py` |
| Advanced Gallery 生成 | `python scripts/generate_advanced_dashboard_gallery.py` |
| Scenario GIF 生成 | `python scripts/generate_scenario_gallery_gif.py` |
| Advanced GIF 生成 | `python scripts/generate_advanced_gallery_gif.py` |
| Evidence GIF 生成 | `python scripts/generate_advanced_evidence_gif.py` |
| デモ GIF 生成 | `python scripts/generate_calibration_demo.py` |

### タスクランナー

```bash
make test
make lint
make app
```

---

## モデル定義と前提

### 量子圧縮パラメータ

```math
r = \eta\sqrt{P}
```

### 損失モデル

(コード変数は `loss_dB`、数式は $\mathrm{loss}_{\mathrm{dB}}$ 表記)

```math
T = 10^{-\frac{\mathrm{loss}_{\mathrm{dB}}}{10}}
```

```math
\hat{a}_{\text{out}} = \sqrt{T}\,\hat{a}_{\text{in}} + \sqrt{1-T}\,\hat{a}_{\text{vac}}
```

---

## テスト & CI

```bash
pip install -e ".[test]"
python -m pytest -q
```

---

## プロジェクト構成

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
