# 양자 광학 버스 - 캘리브레이션 대시보드

**Languages:** [English](../README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [中文](README.zh.md)

<i lang="en">One Waveguide (Hardware), Infinite States (Software)</i> 기반 하이브리드 양자-고전 시뮬레이터입니다.

---

## 실시간 데모

`calibration_demo.gif` 는 포트 전력 0–200 mW 스윕과 손실 0–2 dB 스윕을 보여줍니다.

<p align="center"><img src="../assets/calibration_demo.gif" width="950" alt="Live demo animation" /></p>

> **Figure 1: 실시간 캘리브레이션 시뮬레이션.**  
> <i lang="en">Intrinsic squeezing (pre-loss)</i>는 입력 전력에 따라 결정되고, <i lang="en">Observed squeezing (post-loss)</i>는 손실에 따라 감소합니다.

---

## 아키텍처

```mermaid
flowchart LR
    UI["<i lang=\"en\">calibration_app.py</i><br/>입력: P, θ, loss_dB/cm, length"]
    HW["<i lang=\"en\">hardware.py</i><br/>모드 프로필 + n<sub>eff</sub>"]
    I["<i lang=\"en\">interface.py</i><br/>P -> r"]
    U["<i lang=\"en\">units.py</i><br/>total_loss_dB -> eta_loss"]
    Q["<i lang=\"en\">quantum.py</i><br/><i lang=\"en\">run_single_mode</i>"]
    MM["<i lang=\"en\">multimode.py</i><br/><i lang=\"en\">run_multimode</i>"]
    TP["<i lang=\"en\">tdm_topology.py</i><br/><i lang=\"en\">simulate_topology</i>"]
    ES["<i lang=\"en\">estimation.py</i><br/><i lang=\"en\">fit_eta_and_loss</i>"]
    CT["<i lang=\"en\">control.py</i><br/><i lang=\"en\">apply_feedback_with_latency</i>"]
    O["<i lang=\"en\">calibration_app.py</i><br/>시각화 + 지표"]

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

## 시나리오 갤러리

| 시나리오 | 이미지 |
|---|---|
| 1. Vacuum Baseline (P = 0 mW) | ![Vacuum Baseline](../assets/dashboard_vacuum.png) |
| 2. Squeezed State (P = 200 mW) | ![Calibration + Squeezing](../assets/dashboard_calibration.png) |
| 3. Decoherence (Pure vs Lossy) | ![Decoherence Comparison](../assets/dashboard_decoherence.png) |

<details>
<summary>시나리오 GIF</summary>
<p align="center"><img src="../assets/scenario_gallery.gif" width="950" alt="Scenario gallery animation" /></p>
</details>

## Advanced 갤러리

| 시나리오 | 이미지 |
|---|---|
| 4. 멀티모드 / 타임빈 | ![Multi-mode Dashboard](../assets/dashboard_multimode.png) |
| 5. 토폴로지 | ![Topology Dashboard](../assets/dashboard_topology.png) |
| 6. 디지털 트윈 + 제어 | ![Digital Twin Dashboard](../assets/dashboard_digital_twin.png) |

<details>
<summary>Advanced GIF</summary>
<p align="center"><img src="../assets/advanced_gallery.gif" width="950" alt="Advanced gallery animation" /></p>
</details>

## Evidence Gallery

<details>
<summary>Evidence GIF</summary>
<p align="center"><img src="../assets/advanced_evidence.gif" width="950" alt="Advanced evidence animation" /></p>
</details>

## 빠른 시작

```bash
pip install -e .
streamlit run src/quantum_optical_bus/calibration_app.py
```

브라우저에서 **http://localhost:8501** 를 열고 슬라이더를 조정하세요.

---

### 추가 명령어

| Task | Command |
|------|---------|
| 갤러리 이미지 생성 | `python scripts/generate_dashboard_gallery.py` |
| Advanced 갤러리 이미지 생성 | `python scripts/generate_advanced_dashboard_gallery.py` |
| Scenario GIF 생성 | `python scripts/generate_scenario_gallery_gif.py` |
| Advanced GIF 생성 | `python scripts/generate_advanced_gallery_gif.py` |
| Evidence GIF 생성 | `python scripts/generate_advanced_evidence_gif.py` |
| 데모 GIF 생성 | `python scripts/generate_calibration_demo.py` |
