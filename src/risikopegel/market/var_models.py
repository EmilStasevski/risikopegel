"""One-day VaR/ES forecasts from a window of past returns.

Conventions: returns in percent; VaR and ES are reported as positive losses (loss = -return), at
confidence level `a` (e.g. 0.99), so the tail probability is p = 1 - a. Every forecaster sees only the
window it is given; the rolling driver (`rolling_forecasts`) passes returns strictly before the forecast day.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from arch import arch_model
from scipy import stats

Arr = npt.NDArray[np.float64]
Forecast = dict[float, tuple[float, float]]  # level -> (VaR, ES)


@dataclass
class State:
    """Per-model warm-start state carried from one day to the next (GARCH starting values)."""

    params: dict[str, Arr] = field(default_factory=dict)


def _empirical(losses: Arr, a: float) -> tuple[float, float]:
    var = float(np.quantile(losses, a))
    es = float(losses[losses >= var].mean())
    return var, es


def historical(r: Arr, levels: Sequence[float], _s: State) -> Forecast:
    return {a: _empirical(-r, a) for a in levels}


def normal(r: Arr, levels: Sequence[float], _s: State) -> Forecast:
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    out = {}
    for a in levels:
        z = stats.norm.ppf(a)
        out[a] = (-mu + sd * z, -mu + sd * stats.norm.pdf(z) / (1 - a))
    return out


def _t_tail(nu: float, a: float) -> tuple[float, float]:
    """VaR and ES (losses) of a standard (not unit-variance) Student-t at level a."""
    q = float(stats.t.ppf(a, nu))
    es = float((nu + q**2) / (nu - 1) * stats.t.pdf(q, nu) / (1 - a))
    return q, es


def student_t(r: Arr, levels: Sequence[float], _s: State) -> Forecast:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nu, loc, scale = stats.t.fit(r)
    nu = max(float(nu), 2.05)
    out = {}
    for a in levels:
        q, es = _t_tail(nu, a)
        out[a] = (-loc + scale * q, -loc + scale * es)
    return out


def ewma(r: Arr, levels: Sequence[float], _s: State, lam: float = 0.94) -> Forecast:
    """RiskMetrics: zero mean, s2_t+1 = lam s2_t + (1-lam) r_t^2, seeded with the window variance."""
    s2 = float(np.var(r[:30]))
    for x in r:
        s2 = lam * s2 + (1 - lam) * float(x) ** 2
    sd = np.sqrt(s2)
    return {a: (sd * stats.norm.ppf(a), sd * stats.norm.pdf(stats.norm.ppf(a)) / (1 - a)) for a in levels}


def _fit_garch(r: Arr, s: State, key: str, o: int) -> tuple[float, float, float, Arr]:
    """Fit (GJR-)GARCH(1,1) with standardized-t innovations; returns mu, sigma_t+1, nu, std. residuals."""
    am = arch_model(r, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist="t", rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # SLSQP convergence warnings; non-finite fits are retried below
        res = am.fit(disp="off", starting_values=s.params.get(key), options={"maxiter": 300})
        if not np.all(np.isfinite(res.params)):  # retry from the library's default start
            res = am.fit(disp="off", options={"maxiter": 300})
    s.params[key] = np.asarray(res.params, dtype=np.float64)
    var1 = float(res.forecast(horizon=1, reindex=False).variance.to_numpy()[-1, 0])
    z = np.asarray(res.std_resid, dtype=np.float64)
    return float(res.params["mu"]), float(np.sqrt(var1)), float(res.params["nu"]), z[np.isfinite(z)]


def _std_t(nu: float, a: float) -> tuple[float, float]:
    q, es = _t_tail(nu, a)
    k = np.sqrt((nu - 2) / nu)  # unit-variance scaling used by arch's standardized t
    return q * k, es * k


def garch_t(r: Arr, levels: Sequence[float], s: State) -> Forecast:
    mu, sd, nu, _ = _fit_garch(r, s, "garch", o=0)
    return {a: tuple(-mu + sd * v for v in _std_t(nu, a)) for a in levels}  # type: ignore[misc]


def gjr_t(r: Arr, levels: Sequence[float], s: State) -> Forecast:
    mu, sd, nu, _ = _fit_garch(r, s, "gjr", o=1)
    return {a: tuple(-mu + sd * v for v in _std_t(nu, a)) for a in levels}  # type: ignore[misc]


def fhs(r: Arr, levels: Sequence[float], s: State) -> Forecast:
    """Filtered historical simulation: GARCH-t filter, empirical quantile of standardized residuals."""
    mu, sd, _, z = _fit_garch(r, s, "garch", o=0)
    return {a: tuple(-mu + sd * v for v in _empirical(-z, a)) for a in levels}  # type: ignore[misc]


def gpd_tail(losses: Arr, a: float, tail_share: float = 0.10) -> tuple[float, float]:
    """Peaks over threshold: GPD fitted to the top `tail_share` of losses (McNeil & Frey 2000)."""
    u = float(np.quantile(losses, 1 - tail_share))
    exc = losses[losses > u] - u
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xi, _, beta = stats.genpareto.fit(exc, floc=0)
    xi = float(np.clip(xi, -0.5, 0.9))
    k_n = exc.size / losses.size
    var = (
        u + beta / xi * (((1 - a) / k_n) ** (-xi) - 1) if abs(xi) > 1e-8 else u - beta * np.log((1 - a) / k_n)
    )
    es = var / (1 - xi) + (beta - xi * u) / (1 - xi)
    return float(var), float(es)


def evt(r: Arr, levels: Sequence[float], s: State) -> Forecast:
    """Conditional EVT: GARCH-t filter, GPD on the standardized-loss tail."""
    mu, sd, _, z = _fit_garch(r, s, "garch", o=0)
    return {a: tuple(-mu + sd * v for v in gpd_tail(-z, a)) for a in levels}  # type: ignore[misc]


MODELS: dict[str, Callable[[Arr, Sequence[float], State], Forecast]] = {
    "historical": historical,
    "normal": normal,
    "student_t": student_t,
    "ewma": ewma,
    "garch_t": garch_t,
    "gjr_t": gjr_t,
    "fhs_garch_t": fhs,
    "evt_garch_t": evt,
}


def rolling_forecasts(
    r: Arr, model: str, *, window: int, levels: Sequence[float], start: int, stop: int | None = None
) -> dict[str, Any]:
    """Forecast day t in [start, stop) from r[t-window:t] only (t itself and later are never seen)."""
    if start < window:
        raise ValueError("start must be >= window")
    stop = r.size if stop is None else stop
    f, s = MODELS[model], State()
    var = {a: np.full(stop - start, np.nan) for a in levels}
    es = {a: np.full(stop - start, np.nan) for a in levels}
    for i, t in enumerate(range(start, stop)):
        fc = f(r[t - window : t], levels, s)
        for a in levels:
            var[a][i], es[a][i] = fc[a]
    return {"var": var, "es": es}
