from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as torch_functional

KNOT_DAYS = np.array([10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730], dtype=np.float64)
STATE_DIM = 14
EPSILON = 1e-12


@dataclass(frozen=True)
class SsviCalibrationConfig:
    adam_steps: int = 50
    lbfgs_steps: int = 75
    adam_lr: float = 5e-2
    lbfgs_lr: float = 0.5
    huber_delta: float = 1e-2
    certification_tolerance: float = 1e-8


@dataclass(frozen=True)
class SsviCalibrationResult:
    raw_state: np.ndarray
    constrained_params: np.ndarray
    fit_rmse_iv: float
    fit_vega_rmse_iv: float
    fit_mae_iv: float
    final_loss: float
    warm_start_used: bool
    node_count: int


def maturity_knots_days() -> np.ndarray:
    return KNOT_DAYS.copy()


def maturity_knots_tau() -> np.ndarray:
    return maturity_knots_days() / 365.0


def _ensure_numpy_2d(values: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array[None, :], True
    return array, False


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def _eta_cap_numpy(theta: np.ndarray, rho: np.ndarray, lam: np.ndarray) -> np.ndarray:
    theta_max = np.maximum(theta[..., -1], EPSILON)
    rho_term = 1.0 + np.abs(rho)
    cap1 = 4.0 / np.maximum(rho_term * np.power(theta_max, 1.0 - lam), EPSILON)
    cap2 = np.sqrt(4.0 / np.maximum(rho_term * np.power(theta_max, 1.0 - 2.0 * lam), EPSILON))
    return np.maximum(np.minimum(cap1, cap2) * 0.999, 1e-6)


def _eta_cap_torch(theta: torch.Tensor, rho: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    theta_max = theta[..., -1].clamp_min(EPSILON)
    rho_term = 1.0 + rho.abs()
    cap1 = 4.0 / (rho_term * torch.pow(theta_max, 1.0 - lam)).clamp_min(EPSILON)
    cap2 = torch.sqrt(
        4.0 / (rho_term * torch.pow(theta_max, 1.0 - 2.0 * lam)).clamp_min(EPSILON)
    )
    floor = torch.tensor(1e-6, device=theta.device, dtype=theta.dtype)
    return torch.maximum(torch.minimum(cap1, cap2) * 0.999, floor)


def _is_constrained_param_array(params: np.ndarray) -> bool:
    if params.shape[-1] != STATE_DIM:
        return False
    theta = params[..., : len(KNOT_DAYS)]
    rho = params[..., len(KNOT_DAYS)]
    eta = params[..., len(KNOT_DAYS) + 1]
    lam = params[..., len(KNOT_DAYS) + 2]
    theta_ok = np.all(theta > 0) and np.all(np.diff(theta, axis=-1) >= -1e-12)
    rho_ok = np.all(np.abs(rho) < 1.0)
    eta_ok = np.all(eta > 0)
    lam_ok = np.all((lam >= 0.0) & (lam <= 0.5))
    return bool(theta_ok and rho_ok and eta_ok and lam_ok)


def raw_to_constrained_params(raw_tensor: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isinstance(raw_tensor, torch.Tensor):
        theta_raw = raw_tensor[..., : len(KNOT_DAYS)]
        rho_raw = raw_tensor[..., len(KNOT_DAYS)]
        eta_raw = raw_tensor[..., len(KNOT_DAYS) + 1]
        lambda_raw = raw_tensor[..., len(KNOT_DAYS) + 2]
        theta = torch.cummax(torch.exp(theta_raw), dim=-1).values
        rho = torch.tanh(rho_raw).clamp(min=-0.999999, max=0.999999)
        lam = (0.5 * torch.sigmoid(lambda_raw)).clamp(min=1e-6, max=0.5 - 1e-6)
        eta_unc = torch.exp(eta_raw)
        eta = torch.minimum(eta_unc, _eta_cap_torch(theta, rho, lam))
        return torch.cat(
            [
                theta,
                rho.unsqueeze(-1),
                eta.unsqueeze(-1),
                lam.unsqueeze(-1),
            ],
            dim=-1,
        )
    raw_array, squeezed = _ensure_numpy_2d(np.asarray(raw_tensor, dtype=np.float64))
    theta = np.maximum.accumulate(np.exp(raw_array[:, : len(KNOT_DAYS)]), axis=1)
    rho = np.clip(np.tanh(raw_array[:, len(KNOT_DAYS)]), -0.999999, 0.999999)
    lam = np.clip(0.5 * (1.0 / (1.0 + np.exp(-raw_array[:, len(KNOT_DAYS) + 2]))), 1e-6, 0.5 - 1e-6)
    eta_unc = np.exp(raw_array[:, len(KNOT_DAYS) + 1])
    eta = np.minimum(eta_unc, _eta_cap_numpy(theta, rho, lam))
    constrained = np.column_stack([theta, rho, eta, lam])
    return constrained[0] if squeezed else constrained


def constrained_to_raw_params(params_tensor: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isinstance(params_tensor, torch.Tensor):
        theta = params_tensor[..., : len(KNOT_DAYS)].clamp_min(EPSILON)
        rho = params_tensor[..., len(KNOT_DAYS)].clamp(min=-0.999999, max=0.999999)
        eta = params_tensor[..., len(KNOT_DAYS) + 1].clamp_min(EPSILON)
        lam = params_tensor[..., len(KNOT_DAYS) + 2].clamp(min=1e-6, max=0.5 - 1e-6)
        return torch.cat(
            [
                torch.log(theta),
                torch.atanh(rho).unsqueeze(-1),
                torch.log(eta).unsqueeze(-1),
                torch.logit((2.0 * lam).clamp(min=1e-6, max=1.0 - 1e-6)).unsqueeze(-1),
            ],
            dim=-1,
        )
    params_array, squeezed = _ensure_numpy_2d(np.asarray(params_tensor, dtype=np.float64))
    raw = np.column_stack(
        [
            np.log(np.maximum(params_array[:, : len(KNOT_DAYS)], EPSILON)),
            np.arctanh(np.clip(params_array[:, len(KNOT_DAYS)], -0.999999, 0.999999)),
            np.log(np.maximum(params_array[:, len(KNOT_DAYS) + 1], EPSILON)),
            _logit(np.clip(2.0 * params_array[:, len(KNOT_DAYS) + 2], 1e-6, 1.0 - 1e-6)),
        ]
    )
    return raw[0] if squeezed else raw


def _theta_curve_torch(tau_query: torch.Tensor, theta_knots: torch.Tensor) -> torch.Tensor:
    knot_tau = torch.as_tensor(maturity_knots_tau(), dtype=tau_query.dtype, device=tau_query.device)
    clamped_tau = tau_query.clamp(min=float(knot_tau[0]), max=float(knot_tau[-1]))
    result = theta_knots[..., -1].unsqueeze(-1).expand_as(clamped_tau)
    for index in range(len(KNOT_DAYS) - 1):
        left_tau = float(knot_tau[index])
        right_tau = float(knot_tau[index + 1])
        weight = (clamped_tau - left_tau) / (right_tau - left_tau)
        segment = theta_knots[..., index].unsqueeze(-1) + (
            theta_knots[..., index + 1].unsqueeze(-1) - theta_knots[..., index].unsqueeze(-1)
        ) * weight
        if index == 0:
            mask = clamped_tau <= right_tau
        else:
            mask = (clamped_tau > left_tau) & (clamped_tau <= right_tau)
        result = torch.where(mask, segment, result)
    return result


def theta_curve(
    tau_query: torch.Tensor | np.ndarray,
    theta_knots: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    if isinstance(tau_query, torch.Tensor) or isinstance(theta_knots, torch.Tensor):
        if not isinstance(tau_query, torch.Tensor):
            tau_query = torch.as_tensor(tau_query, dtype=torch.float64)
        if not isinstance(theta_knots, torch.Tensor):
            theta_knots = torch.as_tensor(theta_knots, dtype=tau_query.dtype, device=tau_query.device)
        return _theta_curve_torch(tau_query, theta_knots)
    tau_array = np.asarray(tau_query, dtype=np.float64)
    theta_array, squeezed = _ensure_numpy_2d(np.asarray(theta_knots, dtype=np.float64))
    knot_tau = maturity_knots_tau()
    flattened = tau_array.reshape(-1)
    curves = [np.interp(flattened, knot_tau, row, left=row[0], right=row[-1]) for row in theta_array]
    stacked = np.vstack(curves).reshape(theta_array.shape[0], *tau_array.shape)
    if squeezed:
        return stacked[0]
    return stacked


def ssvi_total_variance(
    m: torch.Tensor | np.ndarray,
    tau: torch.Tensor | np.ndarray,
    params: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    if isinstance(m, torch.Tensor) or isinstance(tau, torch.Tensor) or isinstance(params, torch.Tensor):
        if not isinstance(m, torch.Tensor):
            m = torch.as_tensor(m, dtype=torch.float64)
        if not isinstance(tau, torch.Tensor):
            tau = torch.as_tensor(tau, dtype=m.dtype, device=m.device)
        if not isinstance(params, torch.Tensor):
            params = torch.as_tensor(params, dtype=m.dtype, device=m.device)
        theta = theta_curve(tau, params[..., : len(KNOT_DAYS)]).clamp_min(EPSILON)
        rho = params[..., len(KNOT_DAYS)].unsqueeze(-1)
        eta = params[..., len(KNOT_DAYS) + 1].unsqueeze(-1).clamp_min(EPSILON)
        lam = params[..., len(KNOT_DAYS) + 2].unsqueeze(-1)
        phi = eta * torch.pow(theta, -lam)
        a = phi * m + rho
        radical = torch.sqrt(a.square() + 1.0 - rho.square())
        return 0.5 * theta * (1.0 + rho * phi * m + radical)
    m_array = np.asarray(m, dtype=np.float64)
    tau_array = np.asarray(tau, dtype=np.float64)
    params_array = np.asarray(params, dtype=np.float64)
    theta = np.maximum(theta_curve(tau_array, params_array[..., : len(KNOT_DAYS)]), EPSILON)
    rho = np.expand_dims(params_array[..., len(KNOT_DAYS)], axis=-1)
    eta = np.expand_dims(np.maximum(params_array[..., len(KNOT_DAYS) + 1], EPSILON), axis=-1)
    lam = np.expand_dims(params_array[..., len(KNOT_DAYS) + 2], axis=-1)
    phi = eta * np.power(theta, -lam)
    a = phi * m_array + rho
    radical = np.sqrt(a**2 + 1.0 - rho**2)
    return 0.5 * theta * (1.0 + rho * phi * m_array + radical)


def ssvi_implied_vol(
    m: torch.Tensor | np.ndarray,
    tau: torch.Tensor | np.ndarray,
    params: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    total_variance = ssvi_total_variance(m, tau, params)
    if isinstance(total_variance, torch.Tensor):
        tau_tensor = tau if isinstance(tau, torch.Tensor) else torch.as_tensor(tau, dtype=total_variance.dtype, device=total_variance.device)
        return torch.sqrt(total_variance / tau_tensor.clamp_min(EPSILON))
    tau_array = np.maximum(np.asarray(tau, dtype=np.float64), EPSILON)
    return np.sqrt(np.asarray(total_variance, dtype=np.float64) / tau_array)


def _butterfly_g(
    m: torch.Tensor | np.ndarray,
    tau: torch.Tensor | np.ndarray,
    params: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    if isinstance(m, torch.Tensor) or isinstance(tau, torch.Tensor) or isinstance(params, torch.Tensor):
        if not isinstance(m, torch.Tensor):
            m = torch.as_tensor(m, dtype=torch.float64)
        if not isinstance(tau, torch.Tensor):
            tau = torch.as_tensor(tau, dtype=m.dtype, device=m.device)
        if not isinstance(params, torch.Tensor):
            params = torch.as_tensor(params, dtype=m.dtype, device=m.device)
        theta = theta_curve(tau, params[..., : len(KNOT_DAYS)]).clamp_min(EPSILON)
        rho = params[..., len(KNOT_DAYS)].unsqueeze(-1)
        eta = params[..., len(KNOT_DAYS) + 1].unsqueeze(-1).clamp_min(EPSILON)
        lam = params[..., len(KNOT_DAYS) + 2].unsqueeze(-1)
        phi = eta * torch.pow(theta, -lam)
        a = phi * m + rho
        radical = torch.sqrt(a.square() + 1.0 - rho.square())
        w = 0.5 * theta * (1.0 + rho * phi * m + radical)
        w_k = 0.5 * theta * phi * (rho + a / radical)
        w_kk = 0.5 * theta * phi.square() * (1.0 - rho.square()) / torch.pow(radical, 3.0)
        term_one = torch.square(1.0 - (m * w_k) / (2.0 * w.clamp_min(EPSILON)))
        term_two = 0.25 * torch.square(w_k) * (1.0 / w.clamp_min(EPSILON) + 0.25)
        return term_one - term_two + 0.5 * w_kk
    m_array = np.asarray(m, dtype=np.float64)
    tau_array = np.asarray(tau, dtype=np.float64)
    params_array = np.asarray(params, dtype=np.float64)
    theta = np.maximum(theta_curve(tau_array, params_array[..., : len(KNOT_DAYS)]), EPSILON)
    rho = np.expand_dims(params_array[..., len(KNOT_DAYS)], axis=-1)
    eta = np.expand_dims(np.maximum(params_array[..., len(KNOT_DAYS) + 1], EPSILON), axis=-1)
    lam = np.expand_dims(params_array[..., len(KNOT_DAYS) + 2], axis=-1)
    phi = eta * np.power(theta, -lam)
    a = phi * m_array + rho
    radical = np.sqrt(a**2 + 1.0 - rho**2)
    w = 0.5 * theta * (1.0 + rho * phi * m_array + radical)
    w_k = 0.5 * theta * phi * (rho + a / radical)
    w_kk = 0.5 * theta * (phi**2) * (1.0 - rho**2) / (radical**3)
    term_one = (1.0 - (m_array * w_k) / (2.0 * np.maximum(w, EPSILON))) ** 2
    term_two = 0.25 * (w_k**2) * (1.0 / np.maximum(w, EPSILON) + 0.25)
    return term_one - term_two + 0.5 * w_kk


def _default_tau_certification_grid() -> np.ndarray:
    knot_days = maturity_knots_days()
    midpoint_days = 0.5 * (knot_days[:-1] + knot_days[1:])
    return np.sort(np.unique(np.concatenate([knot_days, midpoint_days]))) / 365.0


def static_arb_certification(
    params: torch.Tensor | np.ndarray,
    m_grid: np.ndarray | None = None,
    tau_grid: np.ndarray | None = None,
    tol: float = 1e-8,
) -> dict[str, Any]:
    if isinstance(params, torch.Tensor):
        raw_array = params.detach().cpu().numpy().astype(np.float64)
    else:
        raw_array = np.asarray(params, dtype=np.float64)
    constrained_array = (
        raw_array if _is_constrained_param_array(raw_array) else np.asarray(raw_to_constrained_params(raw_array), dtype=np.float64)
    )
    m_values = np.asarray(
        m_grid if m_grid is not None else np.linspace(np.log(0.60), np.log(2.00), 41),
        dtype=np.float64,
    )
    tau_values = np.asarray(
        tau_grid if tau_grid is not None else _default_tau_certification_grid(),
        dtype=np.float64,
    )
    mesh_m = np.tile(m_values[None, :], (tau_values.shape[0], 1))
    mesh_tau = np.tile(tau_values[:, None], (1, m_values.shape[0]))
    total_variance = np.asarray(
        ssvi_total_variance(mesh_m, mesh_tau, constrained_array),
        dtype=np.float64,
    )
    calendar_residuals = np.diff(total_variance, axis=0)
    butterfly_residuals = np.asarray(
        _butterfly_g(mesh_m, mesh_tau, constrained_array),
        dtype=np.float64,
    )
    max_negative_calendar = (
        float(calendar_residuals.min()) if np.any(calendar_residuals < -tol) else 0.0
    )
    max_negative_butterfly = (
        float(butterfly_residuals.min()) if np.any(butterfly_residuals < -tol) else 0.0
    )
    calendar_violations = int(np.sum(calendar_residuals < -tol))
    butterfly_violations = int(np.sum(butterfly_residuals < -tol))
    return {
        "max_negative_calendar_residual": max_negative_calendar,
        "max_negative_butterfly_residual": max_negative_butterfly,
        "calendar_violation_count": calendar_violations,
        "butterfly_violation_count": butterfly_violations,
        "passes_static_arb": calendar_violations == 0 and butterfly_violations == 0,
    }


def initial_params_from_nodes(nodes: pl.DataFrame) -> np.ndarray:
    if nodes.is_empty():
        raise ValueError("Cannot initialize SSVI parameters from an empty node panel.")
    knot_tau = maturity_knots_tau()
    theta_guesses: list[float] = []
    for target_tau in knot_tau:
        candidate = (
            nodes.with_columns(
                (pl.col("tau") - target_tau).abs().alias("tau_distance"),
                pl.col("m").abs().alias("m_abs"),
            )
            .sort(["tau_distance", "m_abs"])
            .head(1)
        )
        if candidate.is_empty():
            raise ValueError("Unable to initialize SSVI knots because no candidate nodes were found.")
        node_iv = float(candidate["node_iv"][0])
        theta_guesses.append(max(target_tau * node_iv * node_iv, 1e-6))
    theta = np.maximum.accumulate(np.asarray(theta_guesses, dtype=np.float64))
    skew_sample = (
        nodes.filter(pl.col("m").abs() <= 0.25)
        .select(["m", "node_iv"])
        .sort("m")
    )
    if skew_sample.height >= 3:
        slope = float(
            np.polyfit(
                skew_sample["m"].to_numpy().astype(np.float64),
                skew_sample["node_iv"].to_numpy().astype(np.float64),
                deg=1,
            )[0]
        )
        rho = float(np.clip(np.tanh(-8.0 * slope), -0.8, 0.8))
    else:
        rho = -0.4
    lam = 0.25
    eta_cap = _eta_cap_numpy(theta[None, :], np.asarray([rho]), np.asarray([lam]))[0]
    eta = min(0.75, 0.8 * eta_cap)
    constrained = np.concatenate([theta, np.asarray([rho, eta, lam], dtype=np.float64)])
    raw = np.asarray(constrained_to_raw_params(constrained), dtype=np.float64)
    return raw


def calibrate_daily_ssvi(
    nodes: pl.DataFrame,
    init_raw: np.ndarray | None,
    config: SsviCalibrationConfig,
) -> SsviCalibrationResult:
    if nodes.is_empty():
        raise ValueError("Cannot calibrate SSVI on an empty node panel.")
    raw_start = (
        np.asarray(init_raw, dtype=np.float64)
        if init_raw is not None
        else initial_params_from_nodes(nodes)
    )
    if raw_start.shape != (STATE_DIM,):
        raise ValueError(
            f"Expected initial SSVI raw state with shape {(STATE_DIM,)}, got {raw_start.shape}."
        )
    m_tensor = torch.as_tensor(np.array(nodes["m"].to_numpy(), copy=True), dtype=torch.float64)
    tau_tensor = torch.as_tensor(np.array(nodes["tau"].to_numpy(), copy=True), dtype=torch.float64)
    iv_tensor = torch.as_tensor(np.array(nodes["node_iv"].to_numpy(), copy=True), dtype=torch.float64)
    vega_tensor = torch.as_tensor(
        np.array(nodes["node_vega"].to_numpy(), copy=True),
        dtype=torch.float64,
    )
    normalized_vega = vega_tensor / vega_tensor.sum().clamp_min(EPSILON)
    raw_parameter = torch.nn.Parameter(torch.as_tensor(raw_start.copy(), dtype=torch.float64))

    def weighted_loss() -> torch.Tensor:
        predicted_iv = ssvi_implied_vol(m_tensor, tau_tensor, raw_to_constrained_params(raw_parameter))
        elementwise = torch_functional.huber_loss(
            predicted_iv,
            iv_tensor,
            delta=config.huber_delta,
            reduction="none",
        )
        return torch.sum(normalized_vega * elementwise)

    adam = torch.optim.Adam([raw_parameter], lr=config.adam_lr)
    for _ in range(config.adam_steps):
        adam.zero_grad(set_to_none=True)
        loss = weighted_loss()
        if not torch.isfinite(loss):
            raise ValueError("SSVI Adam calibration produced a non-finite loss.")
        loss.backward()
        adam.step()

    lbfgs = torch.optim.LBFGS([raw_parameter], lr=config.lbfgs_lr, max_iter=1, line_search_fn="strong_wolfe")
    for _ in range(config.lbfgs_steps):
        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss = weighted_loss()
            if not torch.isfinite(loss):
                raise ValueError("SSVI LBFGS calibration produced a non-finite loss.")
            loss.backward()
            return loss

        loss = lbfgs.step(closure)
        if not torch.isfinite(loss):
            raise ValueError("SSVI LBFGS calibration returned a non-finite loss.")

    with torch.no_grad():
        constrained = raw_to_constrained_params(raw_parameter).detach().cpu().numpy().astype(np.float64)
        canonical_raw = np.asarray(constrained_to_raw_params(constrained), dtype=np.float64)
        fitted_iv = np.asarray(
            ssvi_implied_vol(
                m_tensor,
                tau_tensor,
                torch.as_tensor(constrained, dtype=torch.float64),
            ).cpu(),
            dtype=np.float64,
        )
        errors = fitted_iv - iv_tensor.cpu().numpy()
        final_loss = float(weighted_loss().item())
        return SsviCalibrationResult(
            raw_state=canonical_raw,
            constrained_params=constrained,
            fit_rmse_iv=float(np.sqrt(np.mean(errors**2))),
            fit_vega_rmse_iv=float(np.sqrt(np.sum(normalized_vega.cpu().numpy() * (errors**2)))),
            fit_mae_iv=float(np.mean(np.abs(errors))),
            final_loss=final_loss,
            warm_start_used=init_raw is not None,
            node_count=int(nodes.height),
        )
