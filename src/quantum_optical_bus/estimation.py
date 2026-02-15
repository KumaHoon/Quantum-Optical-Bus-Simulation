"""
Parameter estimation utilities for digital-twin calibration.

MVP target:
- fit_eta_and_loss(data, model=...) -> eta_hat, loss_hat, diagnostics
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv

import numpy as np
from scipy.optimize import least_squares


def _as_array(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return arr


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    rows: dict[str, list[float]] = {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header")
        for field in reader.fieldnames:
            rows[field] = []
        for row in reader:
            for key, value in row.items():
                if value is None or value == "":
                    continue
                rows[key].append(float(value))
    return {k: np.asarray(v, dtype=float) for k, v in rows.items() if len(v) > 0}


def _load_data(data: dict[str, Any] | str | Path) -> dict[str, np.ndarray]:
    if isinstance(data, (str, Path)):
        return _load_csv(Path(data))
    if isinstance(data, dict):
        return {k: _as_array(v, k) for k, v in data.items()}
    raise TypeError("data must be dict, str, or Path")


def _predicted_variances(pump_power_mw: np.ndarray, eta: float, loss_db: float) -> tuple[np.ndarray, np.ndarray]:
    r = eta * np.sqrt(np.clip(pump_power_mw, 0.0, None))
    transmissivity = 10.0 ** (-loss_db / 10.0)
    vac = 0.5
    var_x = transmissivity * (vac * np.exp(-2.0 * r)) + (1.0 - transmissivity) * vac
    var_p = transmissivity * (vac * np.exp(2.0 * r)) + (1.0 - transmissivity) * vac
    return var_x, var_p


def fit_eta_and_loss(
    data: dict[str, Any] | str | Path,
    model: str = "auto",
) -> tuple[float, float, dict[str, Any]]:
    """Fit (eta, loss_db) from measured data using nonlinear least squares.

    Parameters
    ----------
    data
        Dict-like or CSV path with at least:
        - pump_power_mw
        and one of:
        - measured_var_x / measured_var_p
        - measured_squeezing_db
    model
        "auto", "variance", or "squeezing_db".

    Returns
    -------
    eta_hat, loss_hat, diagnostics
    """
    arr = _load_data(data)

    if "pump_power_mw" not in arr:
        raise ValueError("data must include pump_power_mw")
    pump = arr["pump_power_mw"]

    has_var_x = "measured_var_x" in arr
    has_var_p = "measured_var_p" in arr
    has_sq_db = "measured_squeezing_db" in arr

    if model == "auto":
        if has_var_x or has_var_p:
            use_model = "variance"
        elif has_sq_db:
            use_model = "squeezing_db"
        else:
            raise ValueError("data must include variance or squeezing columns")
    else:
        use_model = model

    if use_model == "variance" and not (has_var_x or has_var_p):
        raise ValueError("variance model requires measured_var_x and/or measured_var_p")
    if use_model == "squeezing_db" and not has_sq_db:
        raise ValueError("squeezing_db model requires measured_squeezing_db")

    n = pump.size
    for key, value in arr.items():
        if value.size not in (0, n):
            raise ValueError(f"column {key} length {value.size} does not match pump_power_mw length {n}")

    init_eta = 0.1
    init_loss = float(np.median(arr["estimated_loss_db"])) if "estimated_loss_db" in arr else 1.0

    def residuals(params: np.ndarray) -> np.ndarray:
        eta, loss_db = float(params[0]), float(params[1])
        pred_var_x, pred_var_p = _predicted_variances(pump, eta, loss_db)

        if use_model == "variance":
            parts = []
            if has_var_x:
                parts.append(pred_var_x - arr["measured_var_x"])
            if has_var_p:
                parts.append(pred_var_p - arr["measured_var_p"])
            return np.concatenate(parts)

        # squeezing_db
        pred_sq_db = -10.0 * np.log10(pred_var_x / 0.5)
        return pred_sq_db - arr["measured_squeezing_db"]

    opt = least_squares(
        residuals,
        x0=np.array([init_eta, init_loss], dtype=float),
        bounds=([1e-6, 0.0], [2.0, 40.0]),
    )

    eta_hat = float(opt.x[0])
    loss_hat = float(opt.x[1])
    final_res = residuals(opt.x)
    rmse = float(np.sqrt(np.mean(final_res**2))) if final_res.size > 0 else 0.0

    diagnostics: dict[str, Any] = {
        "success": bool(opt.success),
        "status": int(opt.status),
        "message": str(opt.message),
        "nfev": int(opt.nfev),
        "cost": float(opt.cost),
        "rmse": rmse,
        "model": use_model,
        "n_samples": int(n),
        "initial_guess": {"eta": init_eta, "loss_db": init_loss},
    }
    return eta_hat, loss_hat, diagnostics
