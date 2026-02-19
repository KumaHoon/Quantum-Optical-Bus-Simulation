# 양자 광학 버스 - 캘리브레이션 대시보드

[English](../README.md) | [日本語](README.ja.md) | 한국어 | [中文](README.zh.md)

One Waveguide (Hardware), Infinite States (Software)를 기반으로 한 하이브리드 양자-고전 시뮬레이션입니다.  
펌프 파워를 입력으로 하여 연속변수 양자 상태 압축 계수 $r=\eta\sqrt{P}$와 손실 후 관측량을 매핑하고, 대시보드에서 보정/디지털 트윈 동작을 시각화합니다.

---

## 실시간 시연

`calibration_demo.gif`는 0–200 mW 펌프 스윕(내재)과 0–2 dB 손실 스윕(관측)의 Wigner 윤곽 변화를 보여줍니다.

![Live demo animation](../assets/calibration_demo.gif)

> **Figure 1: 실시간 캘리브레이션 데모.**  
> Intrinsic Squeezing (pre-loss)는 동일한 펌프에서는 거의 고정값이고,  
> Observed Squeezing (post-loss)는 손실이 커질수록 감소합니다.

---

## 아키텍처

아키텍처는 다음 서사로 정리됩니다.

1) 맥락(사용자 + Streamlit UI),  
2) 핵심 핫패스(입력 → 매핑/단위 변환 → 시뮬레이션 → 진단 결과),  
3) 선택 경로(하드웨어/백엔드),  
4) 피드백 경로(`estimation.py` → `control.py`).

```mermaid
flowchart TD
  %% 맥락
  User([Researcher / Operator]) --> App["Streamlit UI<br/>calibration_app.py<br/>orchestrator"]

  %% 핫패스
  subgraph HOT["핫패스: 입력 → 매핑/단위변환 → 시뮬레이션 → 출력"]
    Mapping["interface.py<br/>P → r 매핑 (r=η√P)"]
    UnitMap["units.py<br/>loss(dB) ↔ transmissivity T"]
    Quantum["quantum.py<br/>Sgate / Rgate / LossChannel"]
    Output["Wigner + covariance + metrics"]
  end

  App -->|슬라이더: P, loss, θ, topology| Mapping --> Quantum
  App -->|손실(dB)| UnitMap --> Quantum
  Quantum --> Output --> App

  %% 선택 경로
  subgraph OPT["선택 경로: 하드웨어/엔진"]
    HW["hardware.py<br/>Meep optional / analytical mock"]
    SF["Strawberry Fields<br/>(Gaussian backend)"]
    Meep["Meep<br/>mode profile / Aeff / n_eff"]
  end

  App -->|하드웨어 파라미터| HW
  HW -.->|선택 profile| App
  Quantum -.-> SF
  HW -.-> Meep
  HW -.->|mode 제약| Constraints["mode constraints"]
  Constraints --> Mapping
  Constraints --> UnitMap

  %% 피드백 경로
  subgraph FB["피드백 경로"]
    Est["estimation.py<br/>fit eta & loss"]
    Ctrl["control.py<br/>phase drift + latency feedback"]
  end

  App -->|측정 분산/스퀴징 곡선| Est
  Est -->|eta_hat, loss_hat| Ctrl
  Ctrl -->|residual / correction| App

  %% 확장 경로
  App --> Multi["multimode.py<br/>mode-wise time-bin pipeline"]
  App --> Top["tdm_topology.py<br/>topology + couplings"]
  Multi -->|mode metrics| App
  Top -->|correlations| App
```

`calibration_app.py`는 UI 입력을 받고 계산 모듈을 호출해 Wigner 시각화, 공분산 지표, 제어/적합 진단 결과를 갱신하는 오케스트레이터입니다.

### 책임 표

| 레이어 | 파일 | 책임 |
|---|---|---|
| 하드웨어 | `src/quantum_optical_bus/hardware.py` | Meep 사용 시 모드 추정, 미지원 시 분석형 mock으로 폴백 |
| 매핑 | `src/quantum_optical_bus/interface.py` | 입력 펌프와 $r=\eta\sqrt{P}$ 대응 정의 |
| 단위 | `src/quantum_optical_bus/units.py` | dB 손실→투과율 변환 및 Wigner / covariance 계산용 스케일 지원 |
| 양자 엔진 | `src/quantum_optical_bus/quantum.py` | `run_single_mode`에서 Sgate/Rgate/LossChannel를 구성해 주요 메트릭 산출 |
| 멀티모드 확장 | `src/quantum_optical_bus/multimode.py` | 모드별 독립 time-bin 가우시안 회로 처리 |
| 위상 토폴로지 | `src/quantum_optical_bus/tdm_topology.py` | 설정 기반 BS 결합 시퀀스 및 모드별 손실 적용 |
| 추정 | `src/quantum_optical_bus/estimation.py` | `fit_eta_and_loss`로 η 및 손실 동시 추정 |
| 제어 | `src/quantum_optical_bus/control.py` | 위상 드리프트 + 지연 포함 간단 제어 루프 |
| 오케스트레이터 UI | `src/quantum_optical_bus/calibration_app.py` | 사이드바 입력으로 각 분석 워크플로우 렌더링 |

### 로드맵(소스 기반)

- `hardware.py`는 Meep 연동 기반 경로를 포함하지만, 현재는 전체 실험형 보정 추출 파이프라인까지 대체하지는 않습니다.
- `interface.py`는 겹침 적분 기반 추정치가 아니라 고정된 결합 계수의 현상론적 매핑을 사용합니다.
- `tdm_topology.py`는 현재 정적 결합 리스트 기반 MVP로, 전 주파수/타이밍 동기화 제어를 완전 통합하지 않습니다.

상세 상호작용은 [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)에서 확인하세요.

---

## 하드웨어 인 더 루프 확장

![Hardware-in-the-loop expansion flow](figures/hil_expansion.png)

로드맵은 다음을 목표로 합니다.

- 광학 경로 (laser/OPA/loop/homodyne)
- 제어 경로 (ADC/FPGA/DAC/EOM driver)
- 월드모델 경로 (`estimation.py` → 제어 계수 갱신 → `hdl` 배포)

---

## 시나리오 갤러리

| 시나리오 | 이미지 |
|---|---|
| **1. Vacuum Baseline (P = 0 mW)** | ![Vacuum Baseline](../assets/dashboard_vacuum.png) |
| **2. Squeezed State (P = 200 mW)** | ![Calibration + Squeezing](../assets/dashboard_calibration.png) |
| **3. Decoherence (Pure vs Lossy)** | ![Decoherence Comparison](../assets/dashboard_decoherence.png) |

### 시나리오 GIF

![Scenario gallery animation](../assets/scenario_gallery.gif)

---

## 고급 갤러리

| 시나리오 | 이미지 |
|---|---|
| **4. Multi-mode / Time-bin Simulator** | ![Multi-mode Dashboard](../assets/dashboard_multimode.png) |
| **5. Topology Simulator** | ![Topology Dashboard](../assets/dashboard_topology.png) |
| **6. Digital Twin + Control** | ![Digital Twin Dashboard](../assets/dashboard_digital_twin.png) |

### 고급 GIF

![Advanced gallery animation](../assets/advanced_gallery.gif)

---

## Evidence 갤러리

### Evidence GIF

![Advanced evidence animation](../assets/advanced_evidence.gif)

---

## 빠른 시작

```bash
# Install
pip install -e .

# 대시보드 실행
streamlit run src/quantum_optical_bus/calibration_app.py
```

브라우저에서 **http://localhost:8501**을 열고, 사이드바에서 파라미터를 조정하세요.

### Docker 빠른 시작

Strawberry Fields 및 Python 3.10 환경에서 동작을 확인했습니다.

```bash
docker build .
docker compose up --build
```

### 추가 명령

| Task | Command |
|---|---|
| 갤러리 이미지 생성 | `python scripts/generate_dashboard_gallery.py` |
| Advanced 갤러리 이미지 생성 | `python scripts/generate_advanced_dashboard_gallery.py` |
| Scenario GIF 생성 | `python scripts/generate_scenario_gallery_gif.py` |
| Advanced GIF 생성 | `python scripts/generate_advanced_gallery_gif.py` |
| Evidence GIF 생성 | `python scripts/generate_advanced_evidence_gif.py` |
| 데모 GIF 생성 | `python scripts/generate_calibration_demo.py` |

### 작업 실행 명령

이 저장소는 최소한의 `Makefile`을 포함합니다.

```bash
make test
make lint
make app
```

`make`가 불가능한 경우:

```bash
python -m pytest -q
python -m compileall src tests
streamlit run src/quantum_optical_bus/calibration_app.py
```

---

## 모델 정의 및 가정

### 압축 파라미터와 제어 변수

```math
r = \eta\sqrt{P}
```

### 손실 모델

(코드 변수는 `loss_dB`이지만, 수식에서는 $\mathrm{loss}_{\mathrm{dB}}$로 표기해 underscore 파싱 오류를 피합니다.)

```math
T = 10^{-\mathrm{loss}_{\mathrm{dB}}/10}
```

```math
\hat{a}_{\mathrm{out}} = \sqrt{T}\,\hat{a}_{\mathrm{in}} + \sqrt{1-T}\,\hat{a}_{\mathrm{vac}}
```

---

## 테스트 & CI

GitHub Actions에서 Ubuntu/Windows/macOS가 기본으로 실행됩니다.

```bash
pip install -e ".[test]"
python -m pytest -q
```

---

## 프로젝트 구조

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
