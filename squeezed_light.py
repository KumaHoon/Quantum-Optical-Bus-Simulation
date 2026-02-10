import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Squeezed Light Simulation

    This simulation visualizes the Wigner function of Squeezed Coherent States.

    The Wigner function $W(x, p)$ is a quasi-probability distribution in phase space.

    ### Simulation Parameters
    """)
    return


@app.cell
def _(mo):
    # Sliders
    r_slider = mo.ui.slider(start=0.0, stop=2.0, step=0.1, value=0.0, label="Squeezing Parameter (r)")
    theta_slider = mo.ui.slider(start=0.0, stop=2*3.14159, step=0.1, value=0.0, label="Squeezing Angle (theta)")
    x0_slider = mo.ui.slider(start=-5.0, stop=5.0, step=0.1, value=2.0, label="Displacement x0")
    p0_slider = mo.ui.slider(start=-5.0, stop=5.0, step=0.1, value=0.0, label="Displacement p0")

    mo.md(
        f"""
        Adjust the state parameters:

        {r_slider}
        {theta_slider}
        {x0_slider}
        {p0_slider}
        """
    )
    return p0_slider, r_slider, theta_slider, x0_slider


@app.cell
def _(np, p0_slider, r_slider, theta_slider, x0_slider):
    # Calculation Logic

    # 1. Get Values
    r = r_slider.value
    theta = theta_slider.value
    x0 = x0_slider.value
    p0 = p0_slider.value

    # 2. Grid
    Limit = 6.0
    N = 200
    x = np.linspace(-Limit, Limit, N)
    p = np.linspace(-Limit, Limit, N)
    X, P = np.meshgrid(x, p)

    # 3. Wigner Function Calculation
    # Rotated coordinates
    # X' = (x - x0)cos(theta) + (p - p0)sin(theta)
    # P' = -(x - x0)sin(theta) + (p - p0)cos(theta)

    dX = X - x0
    dP = P - p0

    X_prime = dX * np.cos(theta) + dP * np.sin(theta)
    P_prime = -dX * np.sin(theta) + dP * np.cos(theta)

    # Squeezing variances (vacuum variance = 1/2 convention)
    # Var(X') = 1/2 * e^(-2r)  -> if r>0, x is squeezed
    # Var(P') = 1/2 * e^(2r)   -> anti-squeezed
    # Actually, standard definition: S(r)|0> has Delta X = e^-r / sqrt(2).
    # Gaussian exponent: - x^2 / (2 * sigma^2) 
    # Term 1: - (X')^2 / (2 * (1/2 * e^(-2r))) = - (X')^2 * e^(2r)
    # Term 2: - (P')^2 / (2 * (1/2 * e^(2r)))  = - (P')^2 * e^(-2r)

    W = (1 / np.pi) * np.exp( - (X_prime**2 * np.exp(2*r) + P_prime**2 * np.exp(-2*r)) )

    return Limit, P, W, X, p0, r, theta, x0


@app.cell
def _(Limit, P, W, X, mo, p0, plt, r, theta, x0):
    # Visualization

    fig, ax = plt.subplots(figsize=(6, 6))

    # Contour plot
    c = ax.contourf(X, P, W, levels=20, cmap='RdBu_r')
    # Wait, meshgrid (x, p): X changes along columns (axis 1), P along rows (axis 0).
    # contourf(X, P, W).
    # Let's check meshgrid. X[0,0] is x[0]. X[0,1] is x[1]. Correct.
    # P[0,0] is p[0]. P[1,0] is p[1]. Correct.

    fig.colorbar(c, ax=ax, label='Wigner Function W(x, p)')

    ax.set_xlabel('Position x')
    ax.set_ylabel('Momentum p')
    ax.set_title(f'Squeezed State (r={r:.2f}, theta={theta:.2f})')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(-Limit, Limit)
    ax.set_ylim(-Limit, Limit)
    ax.set_aspect('equal')

    # Add circle for reference (Vacuum noise reference)
    circle = plt.Circle((x0, p0), 1.0, color='black', fill=False, linestyle='--', alpha=0.5, label='Vacuum Variance')
    ax.add_patch(circle)

    plt.tight_layout()

    mo.mpl.interactive(fig)
    return


if __name__ == "__main__":
    app.run()
