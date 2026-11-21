"""Module E: CRR retail IRB formula, Monte Carlo vs closed form, EU IRRBB shocks and floor, EVE sign."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from risikopegel.rtf import credit, irrbb


def test_retail_correlation_bounds() -> None:
    # CRR Art. 154(1): 0.16 for PD -> 0, 0.03 for high PD
    r = credit.retail_other_correlation(np.array([1e-6, 0.5, 1.0]))
    assert r[0] == pytest.approx(0.16, abs=1e-4)
    assert r[2] == pytest.approx(0.03, abs=1e-6)
    assert np.all(np.diff(credit.retail_other_correlation(np.linspace(0.001, 0.3, 50))) < 0)


def test_irb_k_hand_calculation() -> None:
    pd_, lgd = 0.02, 0.45
    w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    r = 0.03 * w + 0.16 * (1 - w)
    g = stats.norm.ppf
    k = lgd * stats.norm.cdf(g(pd_) / np.sqrt(1 - r) + np.sqrt(r / (1 - r)) * g(0.999)) - lgd * pd_
    assert float(credit.irb_retail_k(np.array([pd_]), lgd)[0]) == pytest.approx(k)
    assert float(credit.irb_retail_rw(np.array([pd_]), lgd)[0]) == pytest.approx(12.5 * k)  # no 1.06 (CRR3)


def test_mc_converges_to_asrf_for_granular_book() -> None:
    pds = np.full(2000, 0.03)
    b = credit.make_buckets(pds, np.full(2000, 1.0), 10)
    s = credit.loss_stats(credit.simulate_losses(b, 0.45, 200_000, 1), 0.999)
    k = float(credit.irb_retail_k(np.array([0.03]), 0.45)[0]) * 2000
    assert s["el"] == pytest.approx(0.03 * 0.45 * 2000, rel=0.02)
    assert s["ul"] == pytest.approx(k, rel=0.08)  # granularity add-on for 2000 loans is small


def test_eu_shock_shapes() -> None:
    t = np.array([0.0, 4.0, 1e6])
    assert irrbb.shock(t, "parallel_up") == pytest.approx([0.02] * 3)
    su = irrbb.shock(t, "short_up")
    assert su[0] == pytest.approx(0.025)
    assert su[1] == pytest.approx(0.025 * np.exp(-1))
    assert su[2] == pytest.approx(0.0)
    st = irrbb.shock(t, "steepener")
    assert st[0] == pytest.approx(-0.65 * 0.025)
    assert st[2] == pytest.approx(0.9 * 0.01)


def test_post_shock_floor() -> None:
    t = np.array([0.0, 10.0, 50.0, 80.0])
    assert irrbb.floor(t) == pytest.approx([-0.015, -0.012, 0.0, 0.0])
    base = np.array([-0.02, 0.0, 0.0, 0.0])  # observed rate below the floor is kept
    out = irrbb.apply_shock(t, base, "parallel_down")
    assert out[0] == pytest.approx(-0.02)
    assert out[1] == pytest.approx(-0.012)


def test_long_asset_short_funding_loses_eve_when_rates_rise() -> None:
    ct, cr = np.array([0.25, 30.0]), np.array([0.02, 0.02])
    a = irrbb.cash_flows(1000, [10], [1.0], 0.03)
    d = irrbb.cash_flows(1000, [0.25], [1.0], 0.01)
    eve0 = irrbb.pv(*a, ct, cr) - irrbb.pv(*d, ct, cr)
    eve1 = irrbb.pv(*a, ct, cr + 0.02) - irrbb.pv(*d, ct, cr + 0.02)
    assert eve1 < eve0
