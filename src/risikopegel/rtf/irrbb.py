"""Interest-rate risk in the banking book: EVE under the EU supervisory shocks (Delegated Reg. (EU)
2024/856, verified: EUR 200/250/100 bp, decay 4 years, post-shock floor) and a historical-simulation
distribution of 1-year EVE changes from the ECB AAA curve."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Arr = npt.NDArray[np.float64]

EUR_SHOCKS_BP = {"parallel": 200.0, "short": 250.0, "long": 100.0}  # Annex Part A, EUR
DECAY = 4.0


def shock(t: Arr, scenario: str, sizes: dict[str, float] = EUR_SHOCKS_BP) -> Arr:
    """Rate change in decimal for maturities t (years), Art. 2 of Reg. 2024/856."""
    par, sh, lg = (sizes[k] / 1e4 for k in ("parallel", "short", "long"))
    s_short = sh * np.exp(-t / DECAY)
    s_long = lg * (1 - np.exp(-t / DECAY))
    out = {
        "parallel_up": np.full_like(t, par),
        "parallel_down": np.full_like(t, -par),
        "steepener": -0.65 * np.abs(s_short) + 0.9 * np.abs(s_long),
        "flattener": 0.8 * np.abs(s_short) - 0.6 * np.abs(s_long),
        "short_up": s_short,
        "short_down": -s_short,
    }
    return np.asarray(out[scenario], dtype=np.float64)


SCENARIOS = ("parallel_up", "parallel_down", "steepener", "flattener", "short_up", "short_down")


def floor(t: Arr) -> Arr:
    """Art. 3(7): -150 bp at immediate maturity, +3 bp per year, 0% from 50 years."""
    return np.asarray(np.minimum(-0.015 + 0.0003 * t, 0.0), dtype=np.float64)


def apply_shock(t: Arr, base: Arr, scenario: str) -> Arr:
    """Shocked curve with the floor; if the observed rate is already below the floor, it is kept."""
    shocked = base + shock(t, scenario)
    return np.asarray(np.maximum(shocked, np.minimum(floor(t), base)), dtype=np.float64)


def interp_curve(tenors: Arr, rates: Arr, t: Arr) -> Arr:
    return np.asarray(np.interp(t, tenors, rates), dtype=np.float64)


def cash_flows(notional: float, tenors: list[float], shares: list[float], rate: float) -> tuple[Arr, Arr]:
    """Bullet tranches: annual coupons up to each tranche's repricing date, principal at repricing.
    Returns (times, amounts) aggregated on a 0.25-year grid."""
    grid = np.arange(0.25, 30.25, 0.25)
    cf = np.zeros_like(grid)
    for T, s in zip(tenors, shares, strict=True):
        n = notional * s
        for c in np.arange(1.0, T + 1e-9, 1.0):
            cf[np.argmin(np.abs(grid - c))] += n * rate
        if T < 1:
            cf[np.argmin(np.abs(grid - T))] += n * rate * T
        cf[np.argmin(np.abs(grid - T))] += n
    return grid, cf


def pv(times: Arr, cf: Arr, curve_t: Arr, curve_r: Arr) -> float:
    """Continuous compounding, as the ECB Svensson spot rates."""
    r = interp_curve(curve_t, curve_r, times)
    return float(np.sum(cf * np.exp(-r * times)))
