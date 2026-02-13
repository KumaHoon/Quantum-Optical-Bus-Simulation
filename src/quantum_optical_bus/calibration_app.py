"""
Calibration Dashboard — Streamlit Application

A "White Box" presentation tool that exposes the calibration logic
connecting classical FDTD hardware parameters to continuous-variable
quantum states.

Run with:
    streamlit run src/quantum_optical_bus/calibration_app.py
"""

import sys
import pathlib

# ---------------------------------------------------------------------------
# Ensure the package is importable when running via `streamlit run` from the
# project root.  We add *both* ``src/`` (for the installed-package case) and
# the project root (in case it is not installed).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
for _p in (_SRC_DIR, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Compat patches — must come before strawberryfields import
import quantum_optical_bus.compat  # noqa: F401, E402

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st

import strawberryfields as sf
from strawberryfields.ops import Sgate, Rgate, LossChannel

from quantum_optical_bus.hardware import run_hardware_simulation, WaveguideConfig
from quantum_optical_bus.interface import calculate_squeezing

# ──────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TDM Optical Bus — Calibration Dashboard",
    page_icon="🔬",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────
# Custom CSS for a polished, premium look
# ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* --- global --- */
    .block-container {padding-top: 1.5rem;}

    /* sidebar header */
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);}
    [data-testid="stSidebar"] * {color: #c9d1d9 !important;}

    /* metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricValue"] {color: #58a6ff !important; font-size: 1.6rem !important;}
    [data-testid="stMetricLabel"] {color: #8b949e !important;}

    /* formula box */
    .formula-box {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px 28px;
        margin: 8px 0;
    }

    /* section dividers */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR — Experimental Setup
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔧 Experimental Setup")

    # --- 1. Hardware Parameters ---
    st.markdown('<p class="section-label">1 · HARDWARE PARAMETERS (MEEP)</p>', unsafe_allow_html=True)

    n_core_display = 2.21
    st.markdown(f"**Refractive Index** $n_{{\\text{{core}}}}$ = `{n_core_display}`  *(LN @ 1550 nm — fixed)*")

    wg_length_mm = st.slider(
        "Waveguide Length  $L$  (mm)",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        key="wg_length",
    )

    loss_db_cm = st.slider(
        "Propagation Loss  (dB/cm)",
        min_value=0.0,
        max_value=3.0,
        value=0.1,
        step=0.05,
        key="loss",
    )

    st.divider()

    # --- 2. Pump Laser Control ---
    st.markdown('<p class="section-label">2 · PUMP LASER CONTROL</p>', unsafe_allow_html=True)

    pump_power_mw = st.slider(
        "Input Power  $P$  (mW)",
        min_value=0.0,
        max_value=500.0,
        value=100.0,
        step=5.0,
        key="power",
    )

    phase_rad = st.slider(
        "Phase  $\\theta$  (rad)",
        min_value=0.0,
        max_value=float(2 * np.pi),
        value=0.0,
        step=0.05,
        key="phase",
    )

    st.divider()
    st.caption("Built with Strawberry Fields + Meep")

# ══════════════════════════════════════════════════════════════════════
#  DERIVED PHYSICS
# ══════════════════════════════════════════════════════════════════════
# Coupling efficiency η  (phenomenological; η√P → r)
ETA = 0.1  # same value used in interface.py

r_param = calculate_squeezing(pump_power_mw)
intrinsic_squeezing_db = -10 * np.log10(np.exp(-2 * r_param)) if r_param > 0 else 0.0

# Loss model: convert dB/cm + length → transmissivity η_loss ∈ [0, 1]
total_loss_db = loss_db_cm * (wg_length_mm / 10.0)        # mm → cm
eta_loss = 10 ** (-total_loss_db / 10.0)                   # linear transmissivity

# ══════════════════════════════════════════════════════════════════════
#  MAIN AREA — Header
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    """
    # 🔬 TDM Optical Bus — Hardware-to-Quantum Calibration
    **Mapping Classical FDTD Parameters to Continuous-Variable (CV) Quantum States**
    """
)

# ══════════════════════════════════════════════════════════════════════
#  SECTION 1 — Phase 1: The Device
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Phase 1 · The Device — LN Ridge Waveguide")

col_hw_plot, col_hw_info = st.columns([3, 2])

# Run hardware simulation (mock Gaussian mode — Meep fallback is silent)
import io as _io, contextlib as _ctxlib
cfg = WaveguideConfig()
with _ctxlib.redirect_stdout(_io.StringIO()), _ctxlib.redirect_stderr(_io.StringIO()):
    n_eff, mode_area, ez_data, extent = run_hardware_simulation(cfg)

with col_hw_plot:
    fig_hw, ax_hw = plt.subplots(figsize=(5, 4))
    im = ax_hw.imshow(ez_data, extent=extent, cmap="RdBu", origin="lower", aspect="auto")
    ax_hw.set_xlabel("x  (\u03bcm)")
    ax_hw.set_ylabel("y  (\u03bcm)")
    ax_hw.set_title("Fundamental Mode Profile  |Ez|", fontsize=12)
    fig_hw.colorbar(im, ax=ax_hw, fraction=0.046, pad=0.04)
    fig_hw.tight_layout()
    st.pyplot(fig_hw)
    plt.close(fig_hw)

with col_hw_info:
    st.metric("Effective Index  $n_{\\text{eff}}$", f"{n_eff:.3f}")
    st.metric("Mode Area", f"{mode_area:.3f}  μm²")
    st.metric("Core Material", "LiNbO₃  (1550 nm)")
    st.markdown(
        r"""
        > The waveguide geometry is **fixed** — it represents the
        > physical hardware fabricated once.  All dynamic control
        > comes from the pump laser parameters (Phase 2).
        """
    )

# ══════════════════════════════════════════════════════════════════════
#  SECTION 2 — Phase 2: The Calibration Bridge (THE CORE)
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Phase 2 · The Calibration Bridge")

col_physics, col_formula = st.columns(2)

with col_physics:
    st.markdown(
        r"""
        The coupling coefficient $(\eta)$ is a
        **phenomenological placeholder** ($\eta = 0.1$, tuned so that
        100 mW $\to$ $r \approx 1$).  It is **not** derived from
        physical parameters in the current version.

        The squeezing parameter scales with the **square root
        of the pump power** — a direct consequence of the parametric
        down-conversion Hamiltonian:

        $$\hat{H}_{\text{int}} \;\propto\; \chi^{(2)}\,\hat{a}^2 + \text{h.c.}$$

        **Roadmap:** replace $\eta$ with a value computed from
        the FDTD mode overlap integral, $\chi^{(2)}$ nonlinearity,
        and waveguide geometry (Meep hook / overlap integral pipeline
        — not yet wired).
        """
    )

with col_formula:
    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown("**Calculated Squeezing Parameter $(r)$:**")
    st.latex(r"r = \eta \, \sqrt{P}")
    st.latex(
        rf"r = {ETA:.2f} \times \sqrt{{{pump_power_mw:.0f}\;\text{{mW}}}}"
        rf"\;=\; \mathbf{{{r_param:.4f}}}"
    )
    st.markdown("**Intrinsic Squeezing (pre-loss):**")
    st.latex(
        rf"\text{{Intrinsic sq.}} = -10\,\log_{{10}}\!\bigl(e^{{-2r}}\bigr)"
        rf"\;=\; \mathbf{{{intrinsic_squeezing_db:.2f}\;\text{{dB}}}}"
    )
    st.caption("Computed from r only (pump power). Does not include propagation/detection loss.")
    if loss_db_cm > 0:
        st.markdown(f"**Channel transmissivity** $\\eta_{{\\text{{loss}}}}$ = `{eta_loss:.4f}`  "
                    f"({total_loss_db:.2f} dB total loss over {wg_length_mm:.1f} mm)")
    st.markdown("</div>", unsafe_allow_html=True)

# Live P → r curve
st.markdown("#### Power–Squeezing Calibration Curve  *(intrinsic, pre-loss)*")
powers_curve = np.linspace(0, 500, 300)
r_curve = ETA * np.sqrt(powers_curve)
db_curve = -10 * np.log10(np.exp(-2 * r_curve))

fig_cal, ax_cal = plt.subplots(figsize=(8, 3.5))
ax_cal.plot(powers_curve, db_curve, color="#58a6ff", linewidth=2, label=r"$-10\log_{10}(e^{-2r})$")
ax_cal.axvline(pump_power_mw, color="#f97583", linestyle="--", linewidth=1.2, label=f"P = {pump_power_mw:.0f} mW")
ax_cal.axhline(intrinsic_squeezing_db, color="#f97583", linestyle=":", linewidth=0.8, alpha=0.6)
ax_cal.scatter([pump_power_mw], [intrinsic_squeezing_db], color="#f97583", zorder=5, s=60)
ax_cal.set_xlabel("Pump Power  P  (mW)")
ax_cal.set_ylabel("Intrinsic Squeezing (pre-loss)  (dB)")
ax_cal.set_title(r"Calibration Curve:  $r = \eta\sqrt{P}$  (intrinsic, pre-loss)")
ax_cal.legend(loc="lower right")
ax_cal.grid(True, alpha=0.25)
fig_cal.tight_layout()
st.pyplot(fig_cal)
plt.close(fig_cal)

# ══════════════════════════════════════════════════════════════════════
#  SECTION 3 — Phase 3: Quantum Result
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Phase 3 · Quantum Result")

tab_wigner, tab_photon, tab_noise = st.tabs(
    ["🌀  Wigner Function", "📊  Photon Number Distribution", "📉  Noise Variance"]
)

# ---------- helpers ----------
GRID_LIMIT = 4.0
GRID_POINTS = 120
xvec = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_POINTS)


@st.cache_data(show_spinner="Running Strawberry Fields …")
def _run_quantum(r: float, theta: float, eta: float):
    """Run a single-mode Gaussian simulation and return state metrics.

    Returns
    -------
    W : ndarray
        Wigner function on the grid.
    mean_photon : float
        Mean photon number of the output state.
    var_x, var_p : float
        Quadrature variances (ħ = 1 convention).
    observed_sq_db : float
        Observed (post-loss) squeezing in dB, from covariance eigenvalues.
    observed_antisq_db : float
        Observed (post-loss) anti-squeezing in dB.
    """
    prog = sf.Program(1)
    with prog.context as q:
        if r > 0:
            Sgate(r) | q[0]
        if theta != 0:
            Rgate(theta) | q[0]
        if eta < 1.0:
            LossChannel(eta) | q[0]

    eng = sf.Engine("gaussian")
    result = eng.run(prog)
    state = result.state

    W = state.wigner(0, xvec, xvec)
    mean_photon = state.mean_photon(0)[0]

    # Quadrature variances: x = (a + a†)/√2,  p = i(a† - a)/√2
    # For Gaussian states we can read from the covariance matrix.
    cov = state.cov()
    var_x = cov[0, 0] / 2.0   # ħ = 1 convention
    var_p = cov[1, 1] / 2.0

    # Observed squeezing from principal variances of the output state
    V = cov / 2.0  # same normalization as var_x/var_p
    eigvals = np.linalg.eigvalsh(V)
    Vmin = eigvals[0]
    Vmax = eigvals[-1]
    vacuum_var = 0.5
    observed_sq_db = float(-10 * np.log10(Vmin / vacuum_var)) if Vmin > 0 else 0.0
    observed_antisq_db = float(10 * np.log10(Vmax / vacuum_var)) if Vmax > 0 else 0.0

    return W, mean_photon, var_x, var_p, observed_sq_db, observed_antisq_db


W, mean_photon, var_x, var_p, observed_sq_db, observed_antisq_db = _run_quantum(
    r_param, phase_rad, eta_loss
)

# ---------- Tab 1: Wigner Function ----------
with tab_wigner:
    col_wig, col_wig_info = st.columns([3, 1])
    with col_wig:
        fig_w, ax_w = plt.subplots(figsize=(6, 5))
        cf = ax_w.contourf(xvec, xvec, W, levels=60, cmap="RdBu_r")
        ax_w.set_xlabel(r"$x$ (position quadrature)")
        ax_w.set_ylabel(r"$p$ (momentum quadrature)")
        title_parts = f"Wigner Function  (r={r_param:.3f}, \u03b8={phase_rad:.2f})"
        if loss_db_cm > 0:
            title_parts += f"  |  loss={total_loss_db:.2f} dB"
        ax_w.set_title(title_parts, fontsize=11)
        fig_w.colorbar(cf, ax=ax_w, fraction=0.046, pad=0.04)
        ax_w.set_aspect("equal")
        fig_w.tight_layout()
        st.pyplot(fig_w)
        plt.close(fig_w)

    with col_wig_info:
        st.metric("Intrinsic squeezing (pre-loss)", f"{intrinsic_squeezing_db:.2f} dB",
                  help="Computed from r only (pump power). Does not include propagation/detection loss.")
        st.metric("Observed squeezing (post-loss)", f"{observed_sq_db:.2f} dB",
                  help="Computed from output covariance eigenvalues after LossChannel.")
        st.metric("Observed anti-squeezing", f"{observed_antisq_db:.2f} dB",
                  help="Anti-squeezed quadrature of the output state.")
        st.metric("Mean Photon #", f"{mean_photon:.2f}")
        if loss_db_cm > 0:
            st.info(
                "🔍 Loss changes the **observed squeezing** (output state), "
                "not the intrinsic *r* parameter. The Wigner function becomes "
                "more circular as loss increases — this is decoherence."
            )

# ---------- Tab 2: Photon Number Distribution ----------
with tab_photon:
    # For a squeezed vacuum: P(n) is non-zero only for even n.
    max_n = 20
    ns = np.arange(max_n + 1)

    if r_param == 0:
        probs = np.zeros(max_n + 1)
        probs[0] = 1.0
    else:
        # Analytical squeezed vacuum (no loss): P(2k) = (tanh r)^{2k} / (cosh r * C(2k,k) * 4^k)
        # Use numerical from mean_photon for lossy case — approximate via thermal-squeezed model.
        # Simple approach: build from Wigner marginals or use SF state.fock_prob
        # Here we use a quick analytical formula for pure squeezed vacuum:
        tanh_r = np.tanh(r_param)
        cosh_r = np.cosh(r_param)
        probs = np.zeros(max_n + 1)
        for n in range(0, max_n + 1, 2):
            k = n // 2
            from math import comb, factorial
            probs[n] = (factorial(n) / (factorial(k) ** 2 * 4 ** k)) * (tanh_r ** n) / cosh_r
        # If loss > 0 the distribution broadens; approximate by mixing with thermal
        if eta_loss < 1.0:
            n_thermal = mean_photon * (1 - eta_loss)
            thermal = np.array([(n_thermal ** n) / ((1 + n_thermal) ** (n + 1)) for n in ns])
            probs = 0.7 * probs + 0.3 * thermal
            probs /= probs.sum() if probs.sum() > 0 else 1.0

    fig_pn, ax_pn = plt.subplots(figsize=(8, 4))
    colors_pn = ["#58a6ff" if n % 2 == 0 else "#8b949e" for n in ns]
    ax_pn.bar(ns, probs, color=colors_pn, edgecolor="#30363d", linewidth=0.5)
    ax_pn.set_xlabel("Photon number  n")
    ax_pn.set_ylabel("P(n)")
    ax_pn.set_title("Photon Number Distribution")
    ax_pn.set_xticks(ns)
    ax_pn.grid(axis="y", alpha=0.25)
    fig_pn.tight_layout()
    st.pyplot(fig_pn)
    plt.close(fig_pn)
    st.caption("Squeezed vacuum produces photon pairs — only **even** photon numbers are populated (blue bars).")

# ---------- Tab 3: Noise Variance ----------
with tab_noise:
    vacuum_var = 0.5  # shot noise level (ħ = 1)

    col_var_plot, col_var_info = st.columns([3, 1])
    with col_var_plot:
        fig_nv, ax_nv = plt.subplots(figsize=(8, 4))

        # Sweep power for variance curves
        powers_nv = np.linspace(0, 500, 200)
        vars_x = []
        vars_p = []
        for pw in powers_nv:
            rr = ETA * np.sqrt(pw)
            # Analytical: squeezed vacuum variances (pure state, no loss)
            vx = 0.5 * np.exp(-2 * rr)
            vp = 0.5 * np.exp(2 * rr)
            # Simple loss model: var -> eta*var + (1-eta)*0.5
            vx = eta_loss * vx + (1 - eta_loss) * 0.5
            vp = eta_loss * vp + (1 - eta_loss) * 0.5
            vars_x.append(vx)
            vars_p.append(vp)

        ax_nv.plot(powers_nv, vars_x, color="#3fb950", linewidth=2, label="Var(x) \u2014 squeezed")
        ax_nv.plot(powers_nv, vars_p, color="#f97583", linewidth=2, label="Var(p) \u2014 anti-squeezed")
        ax_nv.axhline(vacuum_var, color="#8b949e", linestyle="--", linewidth=1, label="Shot noise limit (vacuum)")
        ax_nv.axvline(pump_power_mw, color="#d2a8ff", linestyle="--", linewidth=1, alpha=0.7)
        ax_nv.scatter([pump_power_mw], [var_x], color="#3fb950", zorder=5, s=50)
        ax_nv.scatter([pump_power_mw], [var_p], color="#f97583", zorder=5, s=50)
        ax_nv.set_xlabel("Pump Power  P  (mW)")
        ax_nv.set_ylabel("Quadrature Variance")
        ax_nv.set_title("Noise Variance vs. Pump Power")
        ax_nv.set_yscale("log")
        ax_nv.legend(loc="upper left")
        ax_nv.grid(True, alpha=0.25)
        fig_nv.tight_layout()
        st.pyplot(fig_nv)
        plt.close(fig_nv)

    with col_var_info:
        delta_x = 10 * np.log10(var_x / vacuum_var)
        delta_p = 10 * np.log10(var_p / vacuum_var)
        st.metric("Var(x)", f"{var_x:.4f}", delta=f"{delta_x:+.1f} dB vs vacuum")
        st.metric("Var(p)", f"{var_p:.4f}", delta=f"{delta_p:+.1f} dB vs vacuum")
        st.markdown(
            r"""
            **Below** the dashed line → noise is *squeezed* below vacuum.
            **Above** the dashed line → noise is *anti-squeezed*.

            Heisenberg relation:
            $$\mathrm{Var}(x)\;\mathrm{Var}(p) \;\geq\; \tfrac{1}{4}$$
            """
        )

# ══════════════════════════════════════════════════════════════════════
#  Footer
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#8b949e; font-size:0.85rem;">
    TDM Optical Bus Calibration Dashboard &nbsp;·&nbsp;
    Strawberry Fields (Gaussian backend) &nbsp;·&nbsp;
    Meep FDTD (analytical mock) &nbsp;·&nbsp;
    Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
