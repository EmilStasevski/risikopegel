"""Credit risk for the retail book: CRR IRB risk weight (Art. 154, verified) and a one-factor
Gaussian-copula Monte Carlo on PD buckets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

Arr = npt.NDArray[np.float64]


def retail_other_correlation(pd_: Arr) -> Arr:
    """CRR Art. 154(1): R = 0.03 w + 0.16 (1 - w), w = (1 - e^(-35 PD)) / (1 - e^(-35))."""
    w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    return np.asarray(0.03 * w + 0.16 * (1 - w), dtype=np.float64)


def irb_retail_k(pd_: Arr, lgd: float | Arr, r: Arr | None = None) -> Arr:
    """Capital requirement per unit EAD, K = RW / 12.5 (CRR Art. 154(1)(ii); no 1.06 factor after CRR3)."""
    r = retail_other_correlation(pd_) if r is None else r
    g = stats.norm.ppf
    cond = stats.norm.cdf(g(pd_) / np.sqrt(1 - r) + np.sqrt(r / (1 - r)) * g(0.999))
    return np.asarray(lgd * cond - lgd * pd_, dtype=np.float64)


def irb_retail_rw(pd_: Arr, lgd: float | Arr) -> Arr:
    return 12.5 * irb_retail_k(pd_, lgd)


@dataclass(frozen=True)
class Buckets:
    pd: Arr  # mean PD per bucket
    n: npt.NDArray[np.int64]  # loans per bucket
    ead: Arr  # total EAD per bucket


def make_buckets(pd_: Arr, ead: Arr, n_buckets: int) -> Buckets:
    order = np.argsort(pd_)
    groups = np.array_split(order, n_buckets)
    return Buckets(
        pd=np.array([pd_[g].mean() for g in groups]),
        n=np.array([g.size for g in groups], dtype=np.int64),
        ead=np.array([ead[g].sum() for g in groups]),
    )


def simulate_losses(b: Buckets, lgd: float, n_sims: int, seed: int, rho: Arr | None = None) -> Arr:
    """Portfolio loss per scenario.

    Systematic factor Z ~ N(0,1); in bucket j, defaults ~ Binomial(n_j, p_j(Z))
    with p_j(Z) = N((G(PD_j) - sqrt(rho_j) Z) / sqrt(1 - rho_j)); loss = defaults * mean EAD_j * LGD."""
    rho = retail_other_correlation(b.pd) if rho is None else rho
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sims)[:, None]
    p = stats.norm.cdf(
        (stats.norm.ppf(b.pd)[None, :] - np.sqrt(rho)[None, :] * z) / np.sqrt(1 - rho)[None, :]
    )
    defaults = rng.binomial(b.n[None, :], p)
    return np.asarray((defaults * (b.ead / b.n)[None, :]).sum(axis=1) * lgd, dtype=np.float64)


def loss_stats(losses: Arr, conf: float) -> dict[str, float]:
    el = float(losses.mean())
    var = float(np.quantile(losses, conf))
    es = float(losses[losses >= var].mean())
    return {"el": el, "var": var, "es": es, "ul": var - el}
