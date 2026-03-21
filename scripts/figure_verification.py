"""Contract-driven figure verifier.

The verifier reads ``docs/FIGURE_CONTRACT.yaml`` and validates generated
figure artifacts with metadata (`.meta.json`) and numeric arrays (`.npz`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image

from scripts.asset_profile import normalize_profiles

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT_DIR / "docs" / "FIGURE_CONTRACT.yaml"


KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "pump_power_mw": ("pump_mw",),
    "quantization_step": ("quantizer_bits",),
    "latency_bins": ("latency_steps",),
    "retention": ("retention_proxy",),
    "intrinsic_sq_db_x": ("intrinsic_sq_db", "intrinsic_squeezing_db"),
    "observed_sq_db_x": ("observed_sq_db",),
    "time_h": ("time_hours",),
    "error_db": ("rmse_db", "error"),
}


@dataclass
class ValidationResult:
    ok: bool
    code: str
    figure_id: str
    profile: str
    validator: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def format(self) -> str:
        if not self.details:
            return f"{self.code} {self.figure_id}/{self.profile} {self.validator}: {self.message}"
        return (
            f"{self.code} {self.figure_id}/{self.profile} {self.validator}: "
            f"{self.message} | {'; '.join(f'{k}={v}' for k, v in self.details.items())}"
        )


@dataclass
class VerificationReport:
    ok: bool
    results: list[ValidationResult]
    summary: dict[str, int] = field(default_factory=dict)


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Contract file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"Invalid contract format: {path}")
    return data


def resolve_target_figures(contract: Mapping[str, Any], target: str) -> list[str]:
    target = "advisor" if target == "mvp" else target
    configured = list(contract.get("targets", {}).get(target, {}).get("figures", []))
    if target in {"advisor", "mvp"}:
        core = [
            "dashboard_vacuum",
            "dashboard_calibration",
            "dashboard_decoherence",
            "sweep_latency",
            "sweep_quantization",
            "calibration_demo",
        ]
        return [fid for fid in configured if fid in core] if configured else core
    if configured:
        return configured
    return [
        fig["id"] for fig in contract.get("figures", []) if isinstance(fig, Mapping) and "id" in fig
    ]


def _resolve_root_path(contract_path: str | Path, output_root: Path) -> Path:
    p = Path(contract_path)
    if p.is_absolute():
        return p
    parts = p.parts
    if not parts:
        return output_root / p
    if parts[0] == "assets":
        return output_root / Path(*parts[1:])
    if parts[0] in {"web", "paper"}:
        return output_root / p
    return output_root / p


def _norm(text: Any) -> str:
    return " ".join(str(text).lower().split())


def _get_key(npz: Mapping[str, Any], key: str) -> str | None:
    if key in npz:
        return key
    for alias in KEY_ALIASES.get(key, ()):
        if alias in npz:
            return alias
    return None


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.loads(f.read())
    if not isinstance(obj, Mapping):
        raise ValueError(f"metadata is not a mapping: {path}")
    return obj


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(v) for k, v in data.items()}


def _validate_outputs(
    figure_id: str, profile: str, outputs: Mapping[str, Any], output_root: Path
) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    key_map = {
        "web": ("web_png", "web_pdf", "web_gif"),
        "paper": ("paper_png", "paper_pdf", "paper_gif"),
    }
    for key in key_map.get(profile, ()):
        if key not in outputs:
            continue
        path = _resolve_root_path(outputs[key], output_root)
        if not path.exists():
            results.append(
                ValidationResult(
                    False,
                    "VAL001",
                    figure_id,
                    profile,
                    "outputs_exist",
                    "missing output",
                    {"path": str(path)},
                )
            )
    return results


def _validate_companions(
    figure_id: str, profile: str, companions: Mapping[str, Any], output_root: Path, needs_npz: bool
) -> tuple[list[ValidationResult], Path | None, Path | None]:
    results: list[ValidationResult] = []
    meta_key = "web_meta" if profile == "web" else "paper_meta"
    npz_key = "web_npz" if profile == "web" else "paper_npz"

    meta_path = None
    npz_path = None

    if meta_key in companions:
        meta_path = _resolve_root_path(companions[meta_key], output_root)
        if not meta_path.exists():
            results.append(
                ValidationResult(
                    False,
                    "VAL002",
                    figure_id,
                    profile,
                    "companions_exist",
                    "missing meta companion",
                    {"path": str(meta_path)},
                )
            )
    if npz_key in companions:
        npz_path = _resolve_root_path(companions[npz_key], output_root)
        if not npz_path.exists():
            # Non-gif figures may require npz; gif-only demo can skip by default.
            if needs_npz:
                results.append(
                    ValidationResult(
                        False,
                        "VAL002",
                        figure_id,
                        profile,
                        "companions_exist",
                        "missing npz companion",
                        {"path": str(npz_path)},
                    )
                )
        else:
            needs_npz = False
    return results, meta_path, npz_path if needs_npz else npz_path


def _validate_meta_schema(
    figure_id: str, profile: str, meta: Mapping[str, Any], required_units: Sequence[str]
) -> list[ValidationResult]:
    required_fields = (
        "figure_id",
        "profile",
        "created_at_utc",
        "generator_script",
        "labels",
        "units",
        "canvas_px",
        "dpi",
    )
    missing = [f for f in required_fields if f not in meta]
    if missing:
        return [
            ValidationResult(
                False,
                "VAL003",
                figure_id,
                profile,
                "meta_schema_valid",
                "missing required keys",
                {"missing": missing},
            )
        ]

    canvas = meta.get("canvas_px")
    dpi = meta.get("dpi")
    expected_canvas = DEFAULT_CANVAS.get(profile)
    expected_dpi = DEFAULT_DPI.get(profile)
    if expected_canvas and tuple(canvas or ()) != tuple(expected_canvas):
        return [
            ValidationResult(
                False,
                "VAL004",
                figure_id,
                profile,
                "canvas_and_dpi_match_profile",
                "canvas mismatch",
                {"expected": expected_canvas, "actual": canvas},
            )
        ]
    if expected_dpi is not None and dpi != expected_dpi:
        return [
            ValidationResult(
                False,
                "VAL004",
                figure_id,
                profile,
                "canvas_and_dpi_match_profile",
                "dpi mismatch",
                {"expected": expected_dpi, "actual": dpi},
            )
        ]

    labels = meta.get("labels") or {}
    units = meta.get("units") or {}
    strings = {_norm(v) for v in list(labels.values()) + list(units.values())}
    for req in required_units:
        if _norm(req) not in strings:
            return [
                ValidationResult(
                    False,
                    "VAL005",
                    figure_id,
                    profile,
                    "units_required_present",
                    "required unit string not found",
                    {"required": req},
                )
            ]
    return []


def no_nan_inf(
    figure_id: str, profile: str, npz: Mapping[str, np.ndarray], keys: Sequence[str], **_
) -> list[ValidationResult]:
    results = []
    for k in keys:
        resolved = _get_key(npz, k)
        if resolved is None:
            results.append(
                ValidationResult(
                    False, "VAL101", figure_id, profile, "no_nan_inf", "missing key", {"key": k}
                )
            )
            continue
        arr = np.asarray(npz[resolved], dtype=float)
        mask = ~np.isfinite(arr)
        if mask.any():
            idx = np.argwhere(mask)[0].tolist()
            results.append(
                ValidationResult(
                    False,
                    "VAL102",
                    figure_id,
                    profile,
                    "no_nan_inf",
                    "contains NaN/Inf",
                    {"key": resolved, "count": int(mask.sum()), "first_index": idx},
                )
            )
    return results


def heisenberg_sanity(
    figure_id: str,
    profile: str,
    npz: Mapping[str, np.ndarray],
    var_x_key: str,
    var_p_key: str,
    vacuum_var: float = 0.5,
    tol: float = 1e-8,
    **_,
) -> list[ValidationResult]:
    kx = _get_key(npz, var_x_key)
    kp = _get_key(npz, var_p_key)
    if kx is None or kp is None:
        missing = [var_x_key if kx is None else None, var_p_key if kp is None else None]
        return [
            ValidationResult(
                False,
                "VAL101",
                figure_id,
                profile,
                "heisenberg_sanity",
                "missing key",
                {"missing": [m for m in missing if m]},
            )
        ]
    lhs = np.asarray(npz[kx], dtype=float) * np.asarray(npz[kp], dtype=float)
    min_val = float(np.nanmin(lhs))
    thr = vacuum_var * vacuum_var
    if min_val < thr - tol:
        return [
            ValidationResult(
                False,
                "VAL110",
                figure_id,
                profile,
                "heisenberg_sanity",
                "product below vacuum bound",
                {"min_product": min_val, "threshold": thr, "tol": tol},
            )
        ]
    return []


def probability_distribution(
    figure_id: str,
    profile: str,
    npz: Mapping[str, np.ndarray],
    p_key: str,
    tol_sum_abs: float = 1e-3,
    require_nonnegative: bool = True,
    **_,
) -> list[ValidationResult]:
    kp = _get_key(npz, p_key)
    if kp is None:
        return [
            ValidationResult(
                False,
                "VAL101",
                figure_id,
                profile,
                "probability_distribution",
                "missing key",
                {"key": p_key},
            )
        ]
    p = np.asarray(npz[kp], dtype=float)
    if require_nonnegative and np.nanmin(p) < -1e-12:
        return [
            ValidationResult(
                False,
                "VAL120",
                figure_id,
                profile,
                "probability_distribution",
                "negative probability",
                {"min": float(np.nanmin(p)), "key": kp},
            )
        ]
    s = float(np.nansum(p))
    if abs(s - 1.0) > tol_sum_abs:
        return [
            ValidationResult(
                False,
                "VAL121",
                figure_id,
                profile,
                "probability_distribution",
                "normalization drift",
                {"sum": s, "abs_error": abs(s - 1.0), "tol": tol_sum_abs},
            )
        ]
    return []


def loss_monotonic_observed(
    figure_id: str,
    profile: str,
    npz: Mapping[str, np.ndarray],
    loss_key: str,
    observed_sq_key: str,
    direction: str = "toward_zero",
    tol_abs: float = 0.02,
    **_,
) -> list[ValidationResult]:
    kl = _get_key(npz, loss_key)
    ko = _get_key(npz, observed_sq_key)
    if kl is None or ko is None:
        return [
            ValidationResult(
                False,
                "VAL101",
                figure_id,
                profile,
                "loss_monotonic_observed",
                "missing key",
                {
                    "missing": [
                        loss_key if kl is None else None,
                        observed_sq_key if ko is None else None,
                    ]
                },
            )
        ]
    loss = np.asarray(npz[kl], dtype=float)
    obs = np.asarray(npz[ko], dtype=float)
    order = np.argsort(loss)
    d = np.abs(obs[order])
    if direction == "toward_zero":
        for i in range(len(d) - 1):
            if d[i + 1] > d[i] + tol_abs:
                return [
                    ValidationResult(
                        False,
                        "VAL130",
                        figure_id,
                        profile,
                        "loss_monotonic_observed",
                        "distance-to-zero increased",
                        {
                            "idx": int(i),
                            "loss": [float(loss[order][i]), float(loss[order][i + 1])],
                            "d": [float(d[i]), float(d[i + 1])],
                        },
                    )
                ]
    return []


def intrinsic_ge_observed(
    figure_id: str,
    profile: str,
    npz: Mapping[str, np.ndarray],
    intrinsic_sq_key: str,
    observed_sq_key: str,
    tol_abs: float = 0.02,
    **_,
) -> list[ValidationResult]:
    kin = _get_key(npz, intrinsic_sq_key)
    ko = _get_key(npz, observed_sq_key)
    if kin is None or ko is None:
        return [
            ValidationResult(
                False,
                "VAL101",
                figure_id,
                profile,
                "intrinsic_ge_observed",
                "missing key",
                {
                    "missing": [
                        intrinsic_sq_key if kin is None else None,
                        observed_sq_key if ko is None else None,
                    ]
                },
            )
        ]
    intrinsic = np.abs(np.asarray(npz[kin], dtype=float))
    observed = np.abs(np.asarray(npz[ko], dtype=float))
    bad = np.where(intrinsic + tol_abs < observed)
    if len(bad[0]) > 0:
        i = int(bad[0][0])
        return [
            ValidationResult(
                False,
                "VAL140",
                figure_id,
                profile,
                "intrinsic_ge_observed",
                "intrinsic weaker than observed",
                {
                    "index": i,
                    "intrinsic": float(intrinsic.flat[i]),
                    "observed": float(observed.flat[i]),
                },
            )
        ]
    return []


def integer_axis(
    figure_id: str, profile: str, npz: Mapping[str, np.ndarray], axis: str, tol: float = 1e-9, **_
) -> list[ValidationResult]:
    k = _get_key(npz, axis)
    if k is None:
        return [
            ValidationResult(
                False, "VAL101", figure_id, profile, "integer_axis", "missing key", {"key": axis}
            )
        ]
    x = np.asarray(npz[k], dtype=float)
    max_err = float(np.nanmax(np.abs(x - np.round(x))))
    if max_err > tol:
        i = int(np.nanargmax(np.abs(x - np.round(x))))
        return [
            ValidationResult(
                False,
                "VAL150",
                figure_id,
                profile,
                "integer_axis",
                "non-integer axis value",
                {"max_deviation": max_err, "index": i, "value": float(x.flat[i])},
            )
        ]
    return []


def matrix_symmetric(
    figure_id: str, profile: str, npz: Mapping[str, np.ndarray], key: str, tol: float = 1e-8, **_
) -> list[ValidationResult]:
    k = _get_key(npz, key)
    if k is None:
        return [
            ValidationResult(
                False, "VAL101", figure_id, profile, "matrix_symmetric", "missing key", {"key": key}
            )
        ]
    a = np.asarray(npz[k], dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        return [
            ValidationResult(
                False,
                "VAL160",
                figure_id,
                profile,
                "matrix_symmetric",
                "not square",
                {"shape": list(a.shape)},
            )
        ]
    err = float(np.nanmax(np.abs(a - a.T)))
    if err > tol:
        return [
            ValidationResult(
                False,
                "VAL161",
                figure_id,
                profile,
                "matrix_symmetric",
                "asymmetry",
                {"max_err": err, "tol": tol},
            )
        ]
    return []


def corr_diagonal_one(
    figure_id: str, profile: str, npz: Mapping[str, np.ndarray], key: str, tol: float = 1e-6, **_
) -> list[ValidationResult]:
    k = _get_key(npz, key)
    if k is None:
        return [
            ValidationResult(
                False,
                "VAL101",
                figure_id,
                profile,
                "corr_diagonal_one",
                "missing key",
                {"key": key},
            )
        ]
    a = np.asarray(npz[k], dtype=float)
    d = np.diag(a)
    err = float(np.nanmax(np.abs(d - 1.0)))
    if err > tol:
        i = int(np.nanargmax(np.abs(d - 1.0)))
        return [
            ValidationResult(
                False,
                "VAL170",
                figure_id,
                profile,
                "corr_diagonal_one",
                "diag != 1",
                {"index": i, "max_err": err, "tol": tol},
            )
        ]
    return []


def fit_reasonable_range(
    figure_id: str,
    profile: str,
    npz: Mapping[str, np.ndarray],
    eta_key: str,
    eta_min: float,
    eta_max: float,
    tol: float = 0.0,
    **_,
) -> list[ValidationResult]:
    k = _get_key(npz, eta_key)
    if k is None:
        return [
            ValidationResult(
                False,
                "VAL101",
                figure_id,
                profile,
                "fit_reasonable_range",
                "missing key",
                {"key": eta_key},
            )
        ]
    v = np.asarray(npz[k], dtype=float)
    mn = float(np.nanmin(v))
    mx = float(np.nanmax(v))
    if mn < eta_min - tol or mx > eta_max + tol:
        return [
            ValidationResult(
                False,
                "VAL180",
                figure_id,
                profile,
                "fit_reasonable_range",
                "out of bounds",
                {"min": mn, "max": mx, "bounds": [eta_min, eta_max]},
            )
        ]
    return []


def trend_degrades(
    figure_id: str,
    profile: str,
    npz: Mapping[str, np.ndarray],
    x_key: str,
    y_key: str,
    expectation: str = "decrease_or_nonincrease",
    tol: float = 1e-6,
    max_violations: int = 0,
    **_,
) -> list[ValidationResult]:
    kx = _get_key(npz, x_key)
    ky = _get_key(npz, y_key)
    if kx is None or ky is None:
        return [
            ValidationResult(
                False,
                "VAL101",
                figure_id,
                profile,
                "trend_degrades",
                "missing key",
                {"missing": [x_key if kx is None else None, y_key if ky is None else None]},
            )
        ]
    x = np.asarray(npz[kx], dtype=float)
    y = np.asarray(npz[ky], dtype=float)
    order = np.argsort(x)
    y = y[order]
    if expectation == "decrease_or_nonincrease":
        viol = np.sum(np.diff(y) > tol)
        if int(viol) > max_violations:
            idx = int(np.where(np.diff(y) > tol)[0][0])
            return [
                ValidationResult(
                    False,
                    "VAL190",
                    figure_id,
                    profile,
                    "trend_degrades",
                    "trend increased",
                    {"index": idx, "violations": int(viol), "max_violations": int(max_violations)},
                )
            ]
    if expectation == "increase_or_nondecrease":
        viol = np.sum(np.diff(y) < -tol)
        if int(viol) > max_violations:
            idx = int(np.where(np.diff(y) < -tol)[0][0])
            return [
                ValidationResult(
                    False,
                    "VAL190",
                    figure_id,
                    profile,
                    "trend_degrades",
                    "trend decreased",
                    {"index": idx, "violations": int(viol), "max_violations": int(max_violations)},
                )
            ]
    return []


def recovery_target(
    figure_id: str,
    profile: str,
    npz: Mapping[str, np.ndarray],
    time_key: str,
    metric_key: str,
    target_time_h: float,
    target_fraction_recovered: float,
    steady_state_max_db: float,
    steady_state_key: str | None = None,
    tail_fraction: float = 0.1,
    **_,
) -> list[ValidationResult]:
    kt = _get_key(npz, time_key)
    km = _get_key(npz, metric_key)
    if kt is None or km is None:
        return [
            ValidationResult(
                False,
                "VAL101",
                figure_id,
                profile,
                "recovery_target",
                "missing key",
                {"missing": [time_key if kt is None else None, metric_key if km is None else None]},
            )
        ]
    t = np.asarray(npz[kt], dtype=float)
    e = np.asarray(npz[km], dtype=float)
    if t.ndim != 1 or e.ndim != 1 or len(t) != len(e) or len(t) == 0:
        return [
            ValidationResult(
                False,
                "VAL200",
                figure_id,
                profile,
                "recovery_target",
                "shape mismatch",
                {"len_time": len(t), "len_metric": len(e)},
            )
        ]
    idx = int(np.argmin(np.abs(t - target_time_h)))
    head = int(max(1, 0.05 * len(e)))
    start = float(np.median(e[:head]))
    current = float(e[idx])
    fraction = (start - current) / max(abs(start), 1e-12)
    if fraction < target_fraction_recovered:
        return [
            ValidationResult(
                False,
                "VAL200",
                figure_id,
                profile,
                "recovery_target",
                "insufficient recovery",
                {
                    "start_error": start,
                    "error_at_target": current,
                    "target_fraction": target_fraction_recovered,
                    "actual_fraction": fraction,
                },
            )
        ]
    ss = None
    if steady_state_key is not None:
        ks = _get_key(npz, steady_state_key)
        if ks is not None:
            ss = float(np.asarray(npz[ks], dtype=float).reshape(-1)[0])
    if ss is None:
        tail_start = max(int((1 - tail_fraction) * len(e)), 0)
        ss = float(np.mean(e[tail_start:]))
    if ss > steady_state_max_db:
        return [
            ValidationResult(
                False,
                "VAL201",
                figure_id,
                profile,
                "recovery_target",
                "steady-state too high",
                {"steady_state": ss, "max": steady_state_max_db},
            )
        ]
    return []


def gif_multiframe(
    figure_id: str,
    profile: str,
    _npz: Mapping[str, np.ndarray],
    web_gif: str,
    min_frames: int = 2,
    **_,
) -> list[ValidationResult]:
    path = Path(web_gif)
    if not path.exists():
        return [
            ValidationResult(
                False,
                "VAL210",
                figure_id,
                profile,
                "gif_multiframe",
                "missing gif",
                {"path": str(path)},
            )
        ]
    with Image.open(path) as im:
        frames = int(getattr(im, "n_frames", 1))
    if frames < min_frames:
        return [
            ValidationResult(
                False,
                "VAL211",
                figure_id,
                profile,
                "gif_multiframe",
                "insufficient frames",
                {"frames": frames, "min_frames": min_frames},
            )
        ]
    return []


HANDLERS = {
    "no_nan_inf": no_nan_inf,
    "heisenberg_sanity": heisenberg_sanity,
    "probability_distribution": probability_distribution,
    "loss_monotonic_observed": loss_monotonic_observed,
    "intrinsic_ge_observed": intrinsic_ge_observed,
    "integer_axis": integer_axis,
    "matrix_symmetric": matrix_symmetric,
    "corr_diagonal_one": corr_diagonal_one,
    "fit_reasonable_range": fit_reasonable_range,
    "trend_degrades": trend_degrades,
    "recovery_target": recovery_target,
    "gif_multiframe": gif_multiframe,
}


def _run_validator(
    validator: str,
    figure_id: str,
    profile: str,
    npz_data: Mapping[str, np.ndarray],
    params: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> list[ValidationResult]:
    handler = HANDLERS.get(validator)
    if handler is None:
        return [
            ValidationResult(
                False,
                "VAL999",
                figure_id,
                profile,
                validator,
                "unknown validator",
                {"validator": validator},
            )
        ]
    if validator == "gif_multiframe":
        return handler(
            figure_id=figure_id,
            profile=profile,
            _npz=npz_data,
            web_gif=outputs.get(f"{profile}_gif", outputs.get("web_gif", "")),
            **params,
        )
    return handler(figure_id=figure_id, profile=profile, npz=npz_data, **params)


def verify_figure(
    figure_id: str,
    profile: str,
    *,
    contract: Mapping[str, Any] | None = None,
    contract_path: Path = DEFAULT_CONTRACT,
    output_root: Path = ROOT_DIR / "assets",
) -> list[ValidationResult]:
    if contract is None:
        contract = load_contract(contract_path)
    figure_map = {f["id"]: f for f in contract.get("figures", []) if isinstance(f, Mapping)}
    if figure_id not in figure_map:
        return [
            ValidationResult(
                False,
                "VAL999",
                figure_id,
                profile,
                "target_resolution",
                "figure missing",
                {"figure_id": figure_id},
            )
        ]

    fig = figure_map[figure_id]
    outputs = fig.get("outputs", {})
    companions = fig.get("companions", {})
    required_units = fig.get("units_required", [])
    validations = fig.get("validations", [])

    results: list[ValidationResult] = []
    results.extend(_validate_outputs(figure_id, profile, outputs, output_root))
    if any(not r.ok for r in results):
        return results

    needs_npz = any(
        v.get("type") != "gif_multiframe" for v in validations if isinstance(v, Mapping)
    )
    c_results, meta_path, npz_path = _validate_companions(
        figure_id, profile, companions, output_root, needs_npz
    )
    results.extend(c_results)
    if meta_path is None or not meta_path.exists():
        return results
    meta = _load_json(meta_path)
    results.extend(_validate_meta_schema(figure_id, profile, meta, required_units))
    if any(not r.ok for r in results):
        return results

    if npz_path is not None and npz_path.exists():
        npz_data = _load_npz(npz_path)
    else:
        npz_data = {}

    for item in validations:
        if not isinstance(item, Mapping):
            continue
        validator = str(item.get("type", ""))
        params = {k: v for k, v in item.items() if k != "type"}
        results.extend(_run_validator(validator, figure_id, profile, npz_data, params, outputs))
    return results


def verify_profile(
    profile: str,
    target: str,
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    output_root: Path = ROOT_DIR / "assets",
    strict: bool = False,
) -> VerificationReport:
    contract = load_contract(contract_path)
    figure_ids = resolve_target_figures(contract, target)
    results: list[ValidationResult] = []
    profiles = normalize_profiles(profile)
    for p in profiles:
        for fid in figure_ids:
            figure_results = verify_figure(
                fid, p, contract=contract, contract_path=contract_path, output_root=output_root
            )
            results.extend(figure_results)
            if strict and any(not r.ok for r in figure_results):
                break
        if strict and any(not r.ok for r in results):
            break
    summary = {
        "passed": sum(1 for r in results if r.ok),
        "failed": sum(1 for r in results if not r.ok),
    }
    return VerificationReport(ok=summary["failed"] == 0, results=results, summary=summary)


DEFAULT_CANVAS = {
    "web": (1600, 1000),
    "paper": (2100, 1350),
}
DEFAULT_DPI = {
    "web": 150,
    "paper": 300,
}
