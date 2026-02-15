"""
Calibration Dashboard â€” Streamlit Application

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

# Compat patches â€” must come before strawberryfields import
import quantum_optical_bus.compat  # noqa: F401, E402

import io
import contextlib
from math import comb, factorial

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st

from quantum_optical_bus.hardware import run_hardware_simulation, WaveguideConfig
from quantum_optical_bus.interface import calculate_squeezing
from quantum_optical_bus.multimode import run_multimode
from quantum_optical_bus.quantum import run_single_mode
from quantum_optical_bus.tdm_topology import simulate_topology
from quantum_optical_bus.estimation import fit_eta_and_loss
from quantum_optical_bus.control import simulate_phase_drift, apply_feedback_with_latency

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Page configuration
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="TDM Optical Bus â€” Calibration Dashboard",
    page_icon="ðŸ”¬",
    layout="wide",
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Custom CSS for a polished, premium look
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SIDEBAR â€” Experimental Setup
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with st.sidebar:
    st.markdown("## ðŸ”§ Experimental Setup")

    # --- 1. Hardware Parameters ---
    st.markdown('<p class="section-label">1 Â· HARDWARE PARAMETERS (MEEP)</p>', unsafe_allow_html=True)

    n_core_display = 2.21
    st.markdown(f"**Refractive Index** $n_{{\\text{{core}}}}$ = `{n_core_display}`  *(LN @ 1550 nm â€” fixed)*")

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
    st.markdown('<p class="section-label">2 Â· PUMP LASER CONTROL</p>', unsafe_allow_html=True)

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

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  DERIVED PHYSICS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Coupling efficiency Î·  (phenomenological; Î·âˆšP â†’ r)
ETA = 0.1  # same value used in interface.py

r_param = calculate_squeezing(pump_power_mw)
intrinsic_squeezing_db = -10 * np.log10(np.exp(-2 * r_param)) if r_param > 0 else 0.0

# Loss model: convert dB/cm + length â†’ transmissivity Î·_loss âˆˆ [0, 1]
total_loss_db = loss_db_cm * (wg_length_mm / 10.0)        # mm â†’ cm
eta_loss = 10 ** (-total_loss_db / 10.0)                   # linear transmissivity

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  MAIN AREA â€” Header
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
st.markdown(
    """
    # ðŸ”¬ TDM Optical Bus â€” Hardware-to-Quantum Calibration
    **Mapping Classical FDTD Parameters to Continuous-Variable (CV) Quantum States**
    """
)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 1 â€” Phase 1: The Device
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
st.markdown("---")
st.markdown("### Phase 1 Â· The Device â€” LN Ridge Waveguide")

col_hw_plot, col_hw_info = st.columns([3, 2])

# Run hardware simulation (mock Gaussian mode â€” Meep fallback is silent)
cfg = WaveguideConfig()
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
    st.metric("Mode Area", f"{mode_area:.3f}  Î¼mÂ²")
    st.metric("Core Material", "LiNbOâ‚ƒ  (1550 nm)")
    st.markdown(
        r"""
        > The waveguide geometry is **fixed** â€” it represents the
        > physical hardware fabricated once.  All dynamic control
        > comes from the pump laser parameters (Phase 2).
        """
    )

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 2 â€” Phase 2: The Calibration Bridge (THE CORE)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
st.markdown("---")
st.markdown("### Phase 2 Â· The Calibration Bridge")

col_physics, col_formula = st.columns(2)

with col_physics:
    st.markdown(
        r"""
        The coupling coefficient $(\eta)$ is a
        **phenomenological placeholder** ($\eta = 0.1$, tuned so that
        100 mW $\to$ $r \approx 1$).  It is **not** derived from
        physical parameters in the current version.

        The squeezing parameter scales with the **square root
        of the pump power** â€” a direct consequence of the parametric
        down-conversion Hamiltonian:

        $$\hat{H}_{\text{int}} \;\propto\; \chi^{(2)}\,\hat{a}^2 + \text{h.c.}$$

        **Roadmap:** replace $\eta$ with a value computed from
        the FDTD mode overlap integral, $\chi^{(2)}$ nonlinearity,
        and waveguide geometry (Meep hook / overlap integral pipeline
        â€” not yet wired).
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

# Live P â†’ r curve
st.markdown("#### Powerâ€“Squeezing Calibration Curve  *(intrinsic, pre-loss)*")
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

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 3 â€” Phase 3: Quantum Result
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
st.markdown("---")
st.markdown("### Phase 3 Â· Quantum Result")

tab_wigner, tab_photon, tab_noise = st.tabs(
    ["ðŸŒ€  Wigner Function", "ðŸ“Š  Photon Number Distribution", "ðŸ“‰  Noise Variance"]
)

# ---------- helpers ----------
GRID_LIMIT = 4.0
GRID_POINTS = 120
xvec = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_POINTS)


@st.cache_data(show_spinner="Running Strawberry Fields â€¦")
def _run_quantum(r: float, theta: float, eta: float):
    """Thin cached wrapper around the shared ``run_single_mode``."""
    res = run_single_mode(r, theta, eta, xvec)
    return res.W, res.mean_photon, res.var_x, res.var_p, res.observed_sq_db, res.observed_antisq_db


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
                "ðŸ” Loss changes the **observed squeezing** (output state), "
                "not the intrinsic *r* parameter. The Wigner function becomes "
                "more circular as loss increases â€” this is decoherence."
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
        # Use numerical from mean_photon for lossy case â€” approximate via thermal-squeezed model.
        # Simple approach: build from Wigner marginals or use SF state.fock_prob
        # Here we use a quick analytical formula for pure squeezed vacuum:
        tanh_r = np.tanh(r_param)
        cosh_r = np.cosh(r_param)
        probs = np.zeros(max_n + 1)
        for n in range(0, max_n + 1, 2):
            k = n // 2
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
    st.caption("Squeezed vacuum produces photon pairs â€” only **even** photon numbers are populated (blue bars).")

# ---------- Tab 3: Noise Variance ----------
with tab_noise:
    vacuum_var = 0.5  # shot noise level (Ä§ = 1)

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
            **Below** the dashed line â†’ noise is *squeezed* below vacuum.
            **Above** the dashed line â†’ noise is *anti-squeezed*.

            Heisenberg relation:
            $$\mathrm{Var}(x)\;\mathrm{Var}(p) \;\geq\; \tfrac{1}{4}$$
            """
        )

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ------------------------------------------------------------------------------
# SECTION 4 - Advanced Simulators / Digital Twin
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Phase 4 - Advanced Simulation + Digital Twin")

tab_mm, tab_topology, tab_twin = st.tabs(
    ["Multi-mode / Time-bin", "Topology Simulator", "Digital Twin Fit + Control"]
)

with tab_mm:
    col_mm_ctrl, col_mm_plot = st.columns([1, 2])

    with col_mm_ctrl:
        mm_bins = st.slider("Number of bins", 1, 12, 4, key="mm_bins")
        mm_base_r = st.slider("Base squeezing r", 0.0, 2.0, float(min(r_param, 2.0)), 0.01, key="mm_base_r")
        mm_phase_step = st.slider(
            "Per-bin rotation step (rad)",
            min_value=float(-np.pi),
            max_value=float(np.pi),
            value=0.2,
            step=0.01,
            key="mm_phase_step",
        )
        mm_end_loss_db = st.slider("End-bin loss (dB)", 0.0, 8.0, 2.0, 0.1, key="mm_end_loss_db")
        mm_wigner_mode = st.slider("Wigner mode index", 0, mm_bins - 1, 0, key="mm_wigner_mode")

    mm_loss_db = np.linspace(0.0, mm_end_loss_db, mm_bins)
    mm_eta = 10 ** (-mm_loss_db / 10.0)
    mm_r = np.full(mm_bins, mm_base_r, dtype=float)
    mm_theta = np.arange(mm_bins, dtype=float) * mm_phase_step
    mm_result = run_multimode(
        r=mm_r,
        theta=mm_theta,
        eta_loss=mm_eta,
        n_modes=mm_bins,
        wigner_mode=mm_wigner_mode,
        xvec=xvec,
    )

    with col_mm_plot:
        fig_mm, axs_mm = plt.subplots(1, 2, figsize=(11, 4.2))
        bins = np.arange(mm_bins)
        axs_mm[0].plot(bins, mm_result.observed_sq_db, marker="o", color="#58a6ff", label="Observed sq (dB)")
        axs_mm[0].plot(
            bins,
            mm_result.observed_antisq_db,
            marker="s",
            color="#f97583",
            label="Observed anti-sq (dB)",
        )
        axs_mm[0].set_xlabel("Time bin index")
        axs_mm[0].set_ylabel("dB")
        axs_mm[0].set_title("Per-bin squeezing metrics")
        axs_mm[0].grid(True, alpha=0.3)
        axs_mm[0].legend(loc="best")

        axs_mm[1].plot(bins, mm_result.var_x, marker="o", color="#3fb950", label="Var(x)")
        axs_mm[1].plot(bins, mm_result.var_p, marker="o", color="#f97583", label="Var(p)")
        axs_mm[1].axhline(0.5, color="#8b949e", ls="--", lw=1, label="Vacuum")
        axs_mm[1].set_xlabel("Time bin index")
        axs_mm[1].set_ylabel("Variance")
        axs_mm[1].set_title("Per-bin quadrature variances")
        axs_mm[1].grid(True, alpha=0.3)
        axs_mm[1].legend(loc="best")
        fig_mm.tight_layout()
        st.pyplot(fig_mm)
        plt.close(fig_mm)

        if mm_result.wigner is not None:
            fig_mm_w, ax_mm_w = plt.subplots(figsize=(4.8, 4.2))
            ax_mm_w.contourf(xvec, xvec, mm_result.wigner, levels=60, cmap="RdBu_r")
            ax_mm_w.set_title(f"Wigner (bin {mm_wigner_mode})")
            ax_mm_w.set_xlabel("x")
            ax_mm_w.set_ylabel("p")
            ax_mm_w.set_aspect("equal")
            fig_mm_w.tight_layout()
            st.pyplot(fig_mm_w)
            plt.close(fig_mm_w)

        st.dataframe(
            {
                "bin": bins,
                "loss_db": mm_loss_db,
                "observed_sq_db": mm_result.observed_sq_db,
                "observed_antisq_db": mm_result.observed_antisq_db,
                "var_x": mm_result.var_x,
                "var_p": mm_result.var_p,
            },
            use_container_width=True,
        )

with tab_topology:
    col_top_ctrl, col_top_plot = st.columns([1, 2])
    with col_top_ctrl:
        top_n = st.slider("Topology modes", 2, 10, 4, key="top_n")
        top_r = st.slider("Per-mode r", 0.0, 1.5, float(min(r_param, 1.5)), 0.01, key="top_r")
        top_phase_step = st.slider(
            "Per-bin phase shift step (rad)",
            min_value=float(-np.pi),
            max_value=float(np.pi),
            value=0.1,
            step=0.01,
            key="top_phase_step",
        )
        top_theta = st.slider("Neighbor coupling theta", 0.0, 1.2, 0.35, 0.01, key="top_theta")
        top_edge_loss_db = st.slider("Per-edge loss (dB)", 0.0, 3.0, 0.2, 0.05, key="top_edge_loss_db")

    edge_eta = 10 ** (-top_edge_loss_db / 10.0)
    top_cfg = {
        "n_modes": top_n,
        "squeezing_r": [top_r] * top_n,
        "phase_shifts": (np.arange(top_n) * top_phase_step).tolist(),
        "loss": [1.0] * top_n,
        "couplings": [
            {"i": i, "j": i + 1, "theta": top_theta, "phi": 0.0, "eta_loss": edge_eta}
            for i in range(top_n - 1)
        ],
    }
    top_result = simulate_topology(top_cfg)

    with col_top_plot:
        fig_top, axs_top = plt.subplots(1, 2, figsize=(11, 4.2))
        im_x = axs_top[0].imshow(top_result.corr_x, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        axs_top[0].set_title("Corr(X) heatmap")
        axs_top[0].set_xlabel("j")
        axs_top[0].set_ylabel("i")
        fig_top.colorbar(im_x, ax=axs_top[0], fraction=0.046, pad=0.04)

        im_p = axs_top[1].imshow(top_result.corr_p, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        axs_top[1].set_title("Corr(P) heatmap")
        axs_top[1].set_xlabel("j")
        axs_top[1].set_ylabel("i")
        fig_top.colorbar(im_p, ax=axs_top[1], fraction=0.046, pad=0.04)
        fig_top.tight_layout()
        st.pyplot(fig_top)
        plt.close(fig_top)

        if top_n > 1:
            fig_nei, ax_nei = plt.subplots(figsize=(8.5, 3.3))
            neighbors = np.arange(top_n - 1)
            ax_nei.plot(neighbors, top_result.neighbor_cov_x, marker="o", color="#58a6ff", label="Cov(x_i,x_{i+1})")
            ax_nei.plot(neighbors, top_result.neighbor_cov_p, marker="o", color="#f97583", label="Cov(p_i,p_{i+1})")
            ax_nei.axhline(0.0, color="#8b949e", ls="--", lw=1)
            ax_nei.set_xlabel("Neighbor pair index i")
            ax_nei.set_ylabel("Covariance")
            ax_nei.set_title("Neighbor correlation metrics")
            ax_nei.grid(True, alpha=0.3)
            ax_nei.legend(loc="best")
            fig_nei.tight_layout()
            st.pyplot(fig_nei)
            plt.close(fig_nei)

with tab_twin:
    col_twin_ctrl, col_twin_plot = st.columns([1, 2])
    with col_twin_ctrl:
        twin_points = st.slider("Synthetic samples", 20, 180, 70, key="twin_points")
        twin_eta_true = st.slider("True eta", 0.03, 0.25, 0.11, 0.001, key="twin_eta_true")
        twin_loss_true = st.slider("True loss (dB)", 0.0, 6.0, 1.8, 0.05, key="twin_loss_true")
        twin_noise_x = st.slider("Noise sigma Var(x)", 0.0, 0.02, 0.003, 0.0005, key="twin_noise_x")
        twin_noise_p = st.slider("Noise sigma Var(p)", 0.0, 0.05, 0.01, 0.001, key="twin_noise_p")
        twin_seed = st.number_input("Random seed", min_value=0, max_value=99999, value=7, step=1, key="twin_seed")
        twin_latency_max = st.slider("Max latency steps", 1, 12, 8, key="twin_latency_max")

    rng = np.random.default_rng(int(twin_seed))
    twin_powers = np.linspace(5.0, 250.0, twin_points)
    twin_r_true = twin_eta_true * np.sqrt(twin_powers)
    twin_trans_true = 10 ** (-twin_loss_true / 10.0)
    twin_var_x_true = twin_trans_true * (0.5 * np.exp(-2.0 * twin_r_true)) + (1.0 - twin_trans_true) * 0.5
    twin_var_p_true = twin_trans_true * (0.5 * np.exp(2.0 * twin_r_true)) + (1.0 - twin_trans_true) * 0.5

    twin_data = {
        "timestamp": np.arange(twin_points, dtype=float),
        "pump_power_mw": twin_powers,
        "measured_var_x": twin_var_x_true + rng.normal(0.0, twin_noise_x, size=twin_points),
        "measured_var_p": twin_var_p_true + rng.normal(0.0, twin_noise_p, size=twin_points),
        "estimated_loss_db": np.full(twin_points, max(twin_loss_true - 0.6, 0.1), dtype=float),
    }
    eta_hat, loss_hat, fit_diag = fit_eta_and_loss(twin_data, model="variance")

    twin_r_hat = eta_hat * np.sqrt(twin_powers)
    twin_trans_hat = 10 ** (-loss_hat / 10.0)
    twin_var_x_hat = twin_trans_hat * (0.5 * np.exp(-2.0 * twin_r_hat)) + (1.0 - twin_trans_hat) * 0.5

    phase_path = simulate_phase_drift(T=260, step_sigma=0.015, drift_rate=0.0015, seed=int(twin_seed) + 11)
    latencies = np.arange(0, twin_latency_max + 1, dtype=int)
    rms_errors = []
    retention = []
    for lat in latencies:
        ctl = apply_feedback_with_latency(
            latency_steps=int(lat),
            true_phase=phase_path,
            measurement_sigma=0.002,
            seed=int(twin_seed) + 17,
        )
        rms_errors.append(ctl["rms_residual_phase_error"])
        retention.append(ctl["mean_retention_proxy"])

    with col_twin_plot:
        m1, m2, m3 = st.columns(3)
        m1.metric("eta (true / fit)", f"{twin_eta_true:.4f} / {eta_hat:.4f}")
        m2.metric("loss dB (true / fit)", f"{twin_loss_true:.3f} / {loss_hat:.3f}")
        m3.metric("fit RMSE", f"{fit_diag['rmse']:.5f}")

        fig_fit, axs_fit = plt.subplots(1, 2, figsize=(11, 4.2))
        axs_fit[0].scatter(twin_powers, twin_data["measured_var_x"], s=16, alpha=0.7, color="#58a6ff", label="Measured Var(x)")
        axs_fit[0].plot(twin_powers, twin_var_x_hat, color="#f97583", lw=2, label="Fitted model Var(x)")
        axs_fit[0].set_xlabel("Pump power (mW)")
        axs_fit[0].set_ylabel("Variance")
        axs_fit[0].set_title("Digital twin fit")
        axs_fit[0].grid(True, alpha=0.3)
        axs_fit[0].legend(loc="best")

        axs_fit[1].plot(latencies, rms_errors, marker="o", color="#f97583", label="RMS residual phase")
        axs_fit[1].plot(latencies, retention, marker="o", color="#3fb950", label="Retention proxy")
        axs_fit[1].set_xlabel("Latency steps")
        axs_fit[1].set_title("Latency vs control quality")
        axs_fit[1].grid(True, alpha=0.3)
        axs_fit[1].legend(loc="best")
        fig_fit.tight_layout()
        st.pyplot(fig_fit)
        plt.close(fig_fit)

# ------------------------------------------------------------------------------
# Footer
#  Footer
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#8b949e; font-size:0.85rem;">
    TDM Optical Bus Calibration Dashboard &nbsp;Â·&nbsp;
    Strawberry Fields (Gaussian backend) &nbsp;Â·&nbsp;
    Meep FDTD (analytical mock) &nbsp;Â·&nbsp;
    Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)


