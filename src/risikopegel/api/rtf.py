"""Economic-perspective recalculation for user-supplied assumptions.

Stdlib only, so the API image stays small.

Credit risk uses the asymptotic single-risk-factor (ASRF/Vasicek) quantile on the aggregate PD buckets from
results/rtf/pd_buckets.csv (no row-level data): UL = sum_j EAD_j LGD [N((G(PD_j) + sqrt(R_j) G(q)) /
sqrt(1 - R_j)) - PD_j], with R_j from CRR Art. 154(1) times a user scaling factor. For q = 0.999 and scale 1
this equals the IRB capital requirement K. The Monte Carlo in scripts/r5_rtf.py adds granularity, which is
negligible for a book of this size.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

N = NormalDist()


def correlation(pd_: float) -> float:
    w = (1 - math.exp(-35 * pd_)) / (1 - math.exp(-35))
    return 0.03 * w + 0.16 * (1 - w)


def load_buckets(path: Path) -> list[tuple[float, float]]:
    with path.open() as f:
        return [(float(r["pd"]), float(r["ead_share"])) for r in csv.DictReader(f)]


def credit_ul(
    buckets: list[tuple[float, float]], ead_total: float, lgd: float, q: float, corr_scale: float
) -> dict[str, float]:
    gq = N.inv_cdf(q)
    ul = el = 0.0
    for pd_, share in buckets:
        r = min(correlation(pd_) * corr_scale, 0.99)
        cond = N.cdf((N.inv_cdf(pd_) + math.sqrt(r) * gq) / math.sqrt(1 - r))
        ead = share * ead_total
        ul += ead * lgd * (cond - pd_)
        el += ead * lgd * pd_
    return {"ul": ul, "el": el}


def economic(params: dict[str, float], buckets: list[tuple[float, float]]) -> dict[str, Any]:
    cr = credit_ul(buckets, params["ead_total"], params["lgd"], params["confidence"], params["corr_scale"])
    ec = {
        "credit": cr["ul"],
        "market": params["market_ec"],
        "irrbb": params["irrbb_ec"],
        "op": params["op_ec"],
    }
    total = sum(ec.values())
    use = total / params["risk_cover"]
    light = "green" if use < 0.8 else ("amber" if use <= 1.0 else "red")
    return {
        "inputs": params,
        "credit_el": cr["el"],
        "ec": ec,
        "total_simple_sum": total,
        "utilisation": use,
        "traffic_light": light,
        "note": "Stylised bank, all inputs are assumptions. Simple-sum aggregation. "
        "Not a regulatory calculation.",
    }
