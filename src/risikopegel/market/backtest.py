"""VaR/ES backtests: Kupiec, Christoffersen, Basel traffic light, Acerbi-Szekely Z2, scoring functions,
Diebold-Mariano.

Inputs: losses L_t (positive = loss, percent), VaR_t and ES_t forecasts made before day t.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

Arr = npt.NDArray[np.float64]


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * float(np.log(y))


def kupiec(hits: npt.NDArray[np.bool_], p: float) -> dict[str, float]:
    """Proportion-of-failures LR test (Kupiec 1995); H0: exception rate = p. chi2(1)."""
    n, x = hits.size, int(hits.sum())
    pi = x / n
    lr = -2 * (_xlogy(n - x, 1 - p) + _xlogy(x, p) - _xlogy(n - x, 1 - pi) - _xlogy(x, pi))
    return {"n": n, "exceptions": x, "rate": pi, "lr_uc": lr, "p_uc": float(stats.chi2.sf(lr, 1))}


def christoffersen(hits: npt.NDArray[np.bool_], p: float) -> dict[str, float]:
    """Independence (first-order Markov) and conditional coverage LR tests (Christoffersen 1998)."""
    h = hits.astype(int)
    prev, cur = h[:-1], h[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    l_alt = _xlogy(n00, 1 - pi0) + _xlogy(n01, pi0) + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1)
    lr_ind = max(-2 * (l_null - l_alt), 0.0)
    lr_uc = kupiec(hits, p)["lr_uc"]
    return {
        "n11": n11,
        "lr_ind": lr_ind,
        "p_ind": float(stats.chi2.sf(lr_ind, 1)),
        "lr_cc": lr_uc + lr_ind,
        "p_cc": float(stats.chi2.sf(lr_uc + lr_ind, 2)),
    }


def traffic_light_zone(exceptions: int, n: int = 250, p: float = 0.01) -> str:
    """Basel zones from the binomial CDF: green below 95%, amber 95% to below 99.99%, red from 99.99%.

    For n=250, p=0.01 this gives green 0-4, amber 5-9, red 10+ (BCBS MAR32 table, verified)."""
    cdf = float(stats.binom.cdf(exceptions, n, p))
    if cdf < 0.95:
        return "green"
    return "amber" if cdf < 0.9999 else "red"


def zone_counts(n: int = 250, p: float = 0.01) -> dict[str, int]:
    """Smallest exception count that starts the amber and the red zone."""
    k = np.arange(n + 1)
    cdf = stats.binom.cdf(k, n, p)
    return {"amber_from": int(k[cdf >= 0.95][0]), "red_from": int(k[cdf >= 0.9999][0])}


def acerbi_szekely_z2(losses: Arr, var: Arr, es: Arr, a: float) -> float:
    """Z2 = 1 - sum(L_t 1{L_t > VaR_t} / ES_t) / (T (1-a)); E[Z2] = 0 if ES is right, < 0 if ES is too low.

    Acerbi & Szekely (2014) suggest the zones Z2 > -0.70 green, -1.8 < Z2 <= -0.70 amber, else red."""
    hit = losses > var
    return float(1 - np.sum(losses[hit] / es[hit]) / (losses.size * (1 - a)))


def z2_zone(z2: float) -> str:
    if z2 > -0.70:
        return "green"
    return "amber" if z2 > -1.8 else "red"


def exceedance_residual_test(losses: Arr, var: Arr, es: Arr, n_boot: int = 5000, seed: int = 0) -> float:
    """McNeil-Frey style: on exception days, residuals L_t - ES_t should have mean 0. One-sided bootstrap
    p-value for mean > 0 (ES too low). NaN if there are fewer than 5 exceptions."""
    hit = losses > var
    e = (losses[hit] - es[hit]) / es[hit]
    if e.size < 5:
        return float("nan")
    rng = np.random.default_rng(seed)
    c = e - e.mean()
    boot = rng.choice(c, size=(n_boot, c.size), replace=True).mean(axis=1)
    return float((np.sum(boot >= e.mean()) + 1) / (n_boot + 1))


def pinball(losses: Arr, var: Arr, a: float) -> Arr:
    """Quantile (tick) loss in return space for the (1-a) return quantile q = -VaR."""
    y, q, tau = -losses, -var, 1 - a
    return np.asarray((tau - (y < q)) * (y - q), dtype=np.float64)


def fz0(losses: Arr, var: Arr, es: Arr, a: float) -> Arr:
    """FZ0 loss for (VaR, ES) jointly (Patton, Ziegel & Chen 2019), return space: v = -VaR, e = -ES < 0."""
    y, v, e, tau = -losses, -var, -es, 1 - a
    return np.asarray(-((y <= v) * (v - y)) / (tau * e) + v / e + np.log(-e) - 1, dtype=np.float64)


def diebold_mariano(l1: Arr, l2: Arr, lag: int | None = None) -> dict[str, float]:
    """DM test on d = l1 - l2 with Newey-West variance. stat < 0: model 1 has the lower loss."""
    d = np.asarray(l1 - l2, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = d.size
    lag = int(np.floor(4 * (n / 100) ** (2 / 9))) if lag is None else lag
    dc = d - d.mean()
    v = float(dc @ dc) / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * float(dc[k:] @ dc[:-k]) / n
    stat = float(d.mean() / np.sqrt(v / n))
    return {"mean_diff": float(d.mean()), "stat": stat, "p_two_sided": float(2 * stats.norm.sf(abs(stat)))}


def summarize(losses: Arr, var: Arr, a: float, es: Arr | None = None) -> dict[str, Any]:
    hits = losses > var
    out: dict[str, Any] = {**kupiec(hits, 1 - a), **christoffersen(hits, 1 - a)}
    out["pinball"] = float(pinball(losses, var, a).mean())
    if es is not None:
        out["z2"] = acerbi_szekely_z2(losses, var, es, a)
        out["z2_zone"] = z2_zone(out["z2"])
        out["p_es_residual"] = exceedance_residual_test(losses, var, es)
        out["fz0"] = float(fz0(losses, var, es, a).mean())
    return out
