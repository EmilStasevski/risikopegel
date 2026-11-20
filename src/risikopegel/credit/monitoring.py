"""Population stability: PSI for scores, CSI (= PSI per feature) for inputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

_EPS = 1e-4  # floor on bin shares so empty bins do not produce infinities


def psi(expected: npt.ArrayLike, actual: npt.ArrayLike, n_bins: int = 10) -> float:
    """Population stability index, bins = quantiles of `expected`; missing values get their own bin.

    Common reading (industry rule of thumb, not a regulation): < 0.1 stable, 0.1-0.25 watch, > 0.25 shift.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e_ok, a_ok = e[~np.isnan(e)], a[~np.isnan(a)]
    edges = np.unique(np.quantile(e_ok, np.linspace(0, 1, n_bins + 1)[1:-1])) if e_ok.size else np.array([])
    e_counts = np.append(np.bincount(np.searchsorted(edges, e_ok, side="right"), minlength=edges.size + 1),
                         np.isnan(e).sum())  # fmt: skip
    a_counts = np.append(np.bincount(np.searchsorted(edges, a_ok, side="right"), minlength=edges.size + 1),
                         np.isnan(a).sum())  # fmt: skip
    pe = np.maximum(e_counts / e.size, _EPS)
    pa = np.maximum(a_counts / a.size, _EPS)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    cats = sorted(set(expected.dropna().unique()) | set(actual.dropna().unique()), key=str)
    pe = np.array([(expected == c).mean() for c in cats] + [expected.isna().mean()])
    pa = np.array([(actual == c).mean() for c in cats] + [actual.isna().mean()])
    pe, pa = np.maximum(pe, _EPS), np.maximum(pa, _EPS)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def csi_table(expected: pd.DataFrame, actual: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for c in expected.columns:
        if expected[c].dtype.kind in "fiub":
            v = psi(expected[c].astype(float), actual[c].astype(float), n_bins)
        else:
            v = psi_categorical(expected[c], actual[c])
        rows.append({"feature": c, "csi": v})
    return pd.DataFrame(rows).sort_values("csi", ascending=False, ignore_index=True)
