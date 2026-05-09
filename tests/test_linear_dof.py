"""Tests for regularization-aware ``_LinearFeature.dof()``.

The plain (unregularized) linear feature contributes one degree of
freedom (the slope :math:`\\beta`). Adding a regularization term shrinks
the *fitted* parameter count: ridge contracts the slope smoothly, L1
and the group-lasso variants soft-threshold it to zero, and Huber
straddles both regimes. These tests pin the analytic limits and an
intermediate fit for each regime so the AIC / BIC / GCV chain that
consumes ``dof()`` stays honest under regularization (issue #79).
"""

from __future__ import annotations

import numpy as np
import pytest

from gamdist.linear_feature import _LinearFeature


def _balanced_x(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=n)


def test_unregularized_dof_is_one() -> None:
    feat = _LinearFeature(name="x")
    feat.initialize(_balanced_x(50))
    feat.optimize(np.zeros(50), rho=1.0)
    assert feat.dof() == pytest.approx(1.0)


def test_l2_dof_shrinks_with_lambda() -> None:
    # Closed-form trace: edof = xtx / (xtx + lambda_2). The feature
    # internally centers x, so we read xtx from the initialized feature
    # rather than computing it from the raw data:
    #   lambda_2 = 0      ->  edof = 1
    #   lambda_2 = xtx    ->  edof = 0.5
    #   lambda_2 -> inf   ->  edof -> 0
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    fpumz = rng.normal(size=200)

    f0 = _LinearFeature(name="x", regularization={"l2": {"coef": 1e-9}})
    f0.initialize(x)
    f0.optimize(fpumz, rho=1.0)
    assert f0.dof() == pytest.approx(1.0, abs=1e-3)
    xtx_centered = float(f0._xtx)

    f_mid = _LinearFeature(name="x", regularization={"l2": {"coef": xtx_centered}})
    f_mid.initialize(x)
    f_mid.optimize(fpumz, rho=1.0)
    assert f_mid.dof() == pytest.approx(0.5, abs=1e-6)

    f_inf = _LinearFeature(name="x", regularization={"l2": {"coef": 1e12}})
    f_inf.initialize(x)
    f_inf.optimize(fpumz, rho=1.0)
    assert f_inf.dof() == pytest.approx(0.0, abs=1e-6)


def test_l1_dof_is_active_set() -> None:
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)

    # Tiny lambda: feature stays active, dof = 1.
    f_small = _LinearFeature(name="x", regularization={"l1": {"coef": 0.01}})
    f_small.initialize(x)
    f_small.optimize(fpumz, rho=1.0)
    assert f_small.dof() == pytest.approx(1.0)

    # Huge lambda: slope soft-thresholded to zero.
    f_huge = _LinearFeature(name="x", regularization={"l1": {"coef": 1e6}})
    f_huge.initialize(x)
    f_huge.optimize(fpumz, rho=1.0)
    assert f_huge._m == pytest.approx(0.0, abs=1e-12)
    assert f_huge.dof() == pytest.approx(0.0)


def test_group_lasso_dof_is_active_set() -> None:
    rng = np.random.default_rng(2)
    n = 200
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)

    f_small = _LinearFeature(name="x", regularization={"group_lasso": {"coef": 0.001}})
    f_small.initialize(x)
    f_small.optimize(fpumz, rho=1.0)
    assert f_small.dof() == pytest.approx(1.0)

    f_huge = _LinearFeature(name="x", regularization={"group_lasso": {"coef": 1e6}})
    f_huge.initialize(x)
    f_huge.optimize(fpumz, rho=1.0)
    assert f_huge._m == pytest.approx(0.0, abs=1e-12)
    assert f_huge.dof() == pytest.approx(0.0)


def test_elastic_net_dof_uses_ridge_trace_when_active() -> None:
    # L1 + L2: when active the trace formula falls back to the L2 piece
    # (the L1 part shrinks the value of m but doesn't change the active
    # Hessian). Match xtx / (xtx + lambda_2).
    rng = np.random.default_rng(3)
    n = 200
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n) + 5.0  # bias makes the active solution likely

    pilot = _LinearFeature(name="x")
    pilot.initialize(x)
    xtx = float(pilot._xtx)

    feat = _LinearFeature(
        name="x",
        regularization={"l1": {"coef": 0.1}, "l2": {"coef": xtx}},
    )
    feat.initialize(x)
    feat.optimize(fpumz, rho=1.0)
    assert abs(feat._m) > 1e-6
    assert feat.dof() == pytest.approx(0.5, abs=1e-6)


def test_huber_in_l2_zone_dof_uses_half_lambda() -> None:
    # When |m| <= delta the half-Huber penalty 0.5 * lambda_h * h(m) is
    # locally 0.5 * lambda_h * m^2, i.e. ridge with strength
    # 0.5 * lambda_h. The hat-matrix denominator therefore picks up
    # 0.5 * lambda_h, and at lambda_h = 2 * xtx the trace must be 0.5.
    rng = np.random.default_rng(4)
    n = 200
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n) * 0.001  # small target -> |m| stays small

    pilot = _LinearFeature(name="x")
    pilot.initialize(x)
    xtx = float(pilot._xtx)

    feat = _LinearFeature(
        name="x",
        regularization={"huber": {"coef": 2.0 * xtx, "delta": 1e6}},
    )
    feat.initialize(x)
    feat.optimize(fpumz, rho=1.0)
    assert abs(feat._m) <= feat._delta_huber  # in the L2 zone
    assert feat.dof() == pytest.approx(0.5, abs=1e-6)


def test_huber_in_l1_zone_dof_falls_back_to_one() -> None:
    # When |m| > delta the huber term is linear and the trace formula
    # collapses to xtx/(xtx + lambda_2). With no L2 component, that's 1.
    rng = np.random.default_rng(5)
    n = 200
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n) + 50.0  # large target -> |m| likely > delta

    feat = _LinearFeature(
        name="x",
        regularization={"huber": {"coef": 0.1, "delta": 0.001}},
    )
    feat.initialize(x)
    feat.optimize(fpumz, rho=1.0)
    assert abs(feat._m) > feat._delta_huber  # in the L1 zone
    assert feat.dof() == pytest.approx(1.0, abs=1e-6)
