# Quantum Optical Bus — Calibration Dashboard

[![CI](https://github.com/KumaHoon/physics_simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/KumaHoon/physics_simulation/actions/workflows/ci.yml)

A hybrid quantum-classical simulation demonstrating **"One Waveguide (Hardware), Infinite States (Software)"** — with a **Calibration Dashboard** that exposes the physics mapping classical FDTD parameters to continuous-variable (CV) quantum states.

✨ Computes the **actual squeezing parameter (*r*)** based on the waveguide's material properties and geometry. The calibration logic is **interactive and transparent**.

---

## 🎬 Live Demo

The dashboard sweeps pump power from 0 → 400 mW, showing the Wigner function evolving from vacuum (circular) to a squeezed state (elliptical) in real-time:

![Demo: Power Sweep](assets/demo_power_sweep.gif)

---

## 🔬 Calibration Dashboard

The dashboard follows a three-phase flow: **Hardware → Calibration → Quantum Result**.

### Phase 1 · The Device (LN Ridge Waveguide)
A Lithium Niobate waveguide at 1550 nm simulated via Meep FDTD (falls back to analytical Gaussian mode).

### Phase 2 · The Calibration Bridge
The core of the presentation — live LaTeX formulas showing:
- **Squeezing parameter:** $r = \eta \sqrt{P}$
- **Squeezing level:** $-10\log_{10}(e^{-2r})$ dB
- Interactive calibration curve with current operating point

### Phase 3 · Quantum Result
Three tabbed visualizations:
- **Wigner Function** — contour plot (becomes "fuzzier" with loss → decoherence)
- **Photon Number Distribution** — even-photon pairing from squeezed vacuum
- **Noise Variance** — squeezed/anti-squeezed quadratures vs shot noise limit

---

## 📸 Scenario Gallery

### 1. Baseline — Vacuum State (P = 0 mW)
![Vacuum Baseline](assets/dashboard_vacuum.png)

### 2. Squeezed State (P = 200 mW)
![Calibration + Squeezing](assets/dashboard_calibration.png)

### 3. Decoherence — Pure vs Lossy
![Decoherence Comparison](assets/dashboard_decoherence.png)

---

## 🚀 How to Run the Simulation

### 🇺🇸 English

1. **Install** the package:
   ```bash
   pip install -e .
   ```
2. **Launch** the Calibration Dashboard:
   ```bash
   streamlit run src/quantum_optical_bus/calibration_app.py
   ```
3. Open **http://localhost:8501** in your browser.
4. Use the **sidebar sliders** to adjust pump power, phase, and loss — watch the quantum state update in real-time.

---

### 🇯🇵 日本語 *(AI-generated translation)*

1. パッケージを**インストール**します：
   ```bash
   pip install -e .
   ```
2. キャリブレーション・ダッシュボードを**起動**します：
   ```bash
   streamlit run src/quantum_optical_bus/calibration_app.py
   ```
3. ブラウザで **http://localhost:8501** を開きます。
4. **サイドバーのスライダー**でポンプ出力・位相・損失を調整し、量子状態がリアルタイムで変化する様子を確認できます。

---

### 🇰🇷 한국어 *(AI 생성 번역)*

1. 패키지를 **설치**합니다:
   ```bash
   pip install -e .
   ```
2. 캘리브레이션 대시보드를 **실행**합니다:
   ```bash
   streamlit run src/quantum_optical_bus/calibration_app.py
   ```
3. 브라우저에서 **http://localhost:8501** 을 엽니다.
4. **사이드바 슬라이더**로 펌프 출력, 위상, 손실을 조정하면 양자 상태가 실시간으로 변화하는 것을 확인할 수 있습니다.

---

### Additional Commands

| Task | Command |
|------|------|
| Legacy Marimo Notebook | `pip install -e ".[full]" && marimo edit src/quantum_optical_bus/app.py` |
| Generate Gallery Images | `python scripts/generate_dashboard_gallery.py` |
| Generate Demo GIF | `python scripts/generate_demo_gif.py` |

---

## 🏗️ Architecture

```
Input (Physics)  →  Calibration (Bridge)  →  Output (Quantum)
   Meep/FDTD           r = η√P              Strawberry Fields
```

| Layer | File | Responsibility |
|-------|------|----------------|
| **Hardware** | `hardware.py` | LN Ridge Waveguide mode simulation (Meep / mock) |
| **Interface** | `interface.py` | Pump power → squeezing parameter mapping |
| **Application** | `application.py` | Quantum Bus model (Strawberry Fields) |
| **Visualization** | `visualization.py` | Matplotlib plotting (BusVisualizer) |
| **Dashboard** | `calibration_app.py` | Streamlit presentation UI |

---

## 🧪 Testing & CI

Tests run on **Ubuntu, Windows, and macOS** via GitHub Actions:

```bash
pip install -e ".[test]"
pytest tests/ -v
```

---

## 📁 Project Structure

```
├── .github/workflows/ci.yml         # CI: Ubuntu / Windows / macOS
├── src/quantum_optical_bus/
│   ├── calibration_app.py            # Streamlit Calibration Dashboard
│   ├── app.py                        # Marimo notebook (legacy UI)
│   ├── hardware.py                   # Layer 1 — Meep / analytical mock
│   ├── interface.py                  # Layer 2 — Power → Squeezing
│   ├── application.py                # Layer 3 — Strawberry Fields
│   ├── visualization.py              # Matplotlib BusVisualizer
│   └── compat.py                     # Dependency patches
├── tests/test_core.py                # Pytest suite (11 tests)
├── scripts/
│   ├── generate_gallery.py           # Original gallery images
│   ├── generate_dashboard_gallery.py # Dashboard scenario images
│   └── generate_demo_gif.py          # Animated demo GIF
└── assets/                           # Generated images & demo
```
