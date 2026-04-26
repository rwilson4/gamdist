"""Tests for the huber regularizer on _LinearFeature.

The penalty ``lambda_h * h_delta(m)`` is a smoothed-L1 / bounded-influence
ridge: quadratic in ``m`` for ``|m| <= delta`` and linear outside that
band. Combined with optional ridge it yields a closed-form 1-D piecewise
solve; combinations with l1 / group_lasso variants are rejected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM
from gamdist.linear_feature import _LinearFeature


def _huber_closed_form(
    b: float, xtx: float, lam_h: float, delta: float, rho: float, lam2: float = 0.0
) -> float:
    denom_lin = xtx + 2.0 * lam2 / rho
    denom_quad = denom_lin + lam_h / rho
    m_quad = b / denom_quad
    if abs(m_quad) <= delta:
        return m_quad
    if m_quad > delta:
        return (b - lam_h * delta / rho) / denom_lin
    return (b + lam_h * delta / rho) / denom_lin


def test_huber_no_coef_raises() -> None:
    with pytest.raises(ValueError, match="No coefficient specified for huber"):
        _LinearFeature(name="x", regularization={"huber": {"delta": 1.0}})


def test_huber_no_delta_raises() -> None:
    with pytest.raises(ValueError, match="No delta specified for huber"):
        _LinearFeature(name="x", regularization={"huber": {"coef": 1.0}})


@pytest.mark.parametrize("bad_delta", [0.0, -0.5])
def test_huber_non_positive_delta_raises(bad_delta: float) -> None:
    with pytest.raises(ValueError, match="huber delta must be positive"):
        _LinearFeature(
            name="x", regularization={"huber": {"coef": 1.0, "delta": bad_delta}}
        )


def test_huber_negative_coef_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _LinearFeature(name="x", regularization={"huber": {"coef": -1.0, "delta": 1.0}})


@pytest.mark.parametrize(
    "other",
    [
        {"l1": {"coef": 0.5}},
        {"group_lasso": {"coef": 0.5}},
        {"group_lasso_inf": {"coef": 0.5}},
    ],
)
def test_huber_rejects_combinations_with_sparsity_penalties(
    other: dict[str, dict[str, float]],
) -> None:
    reg = {"huber": {"coef": 1.0, "delta": 1.0}, **other}
    with pytest.raises(ValueError, match="huber"):
        _LinearFeature(name="x", regularization=reg)


def test_huber_smoothing_scales_lambda() -> None:
    feat = _LinearFeature(
        name="x", regularization={"huber": {"coef": 0.4, "delta": 1.0}}
    )
    feat.initialize(np.array([0.0, 1.0, 2.0, 3.0]), smoothing=2.5)
    assert feat._has_huber
    assert feat._lambda_huber == pytest.approx(1.0)
    assert feat._delta_huber == pytest.approx(1.0)


def test_huber_lambda_zero_matches_unpenalized() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)

    plain = _LinearFeature(name="x")
    plain.initialize(x)
    plain.optimize(fpumz, rho=1.0)

    zero = _LinearFeature(
        name="x", regularization={"huber": {"coef": 0.0, "delta": 1.0}}
    )
    zero.initialize(x)
    zero.optimize(fpumz, rho=1.0)

    assert zero._m == pytest.approx(plain._m, rel=1e-12, abs=1e-12)


def test_huber_quadratic_region_closed_form() -> None:
    # Pick fpumz so the unpenalized slope falls inside the Huber band, where
    # the penalty acts like ridge with strength lambda.
    rng = np.random.default_rng(7)
    n = 80
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)
    rho = 2.0
    lam_h = 0.6
    delta = 5.0  # large enough that |m| <= delta in the quadratic region

    feat = _LinearFeature(
        name="x", regularization={"huber": {"coef": lam_h, "delta": delta}}
    )
    feat.initialize(x)
    feat.optimize(fpumz, rho=rho)

    x_centered = x - x.mean()
    xtx = float(x_centered.dot(x_centered))
    b = float(x_centered.dot(-fpumz))
    expected = _huber_closed_form(b, xtx, lam_h, delta, rho)
    assert abs(expected) <= delta  # confirm we're in the quadratic region
    assert feat._m == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_huber_linear_region_closed_form() -> None:
    # Force the unpenalized slope outside the Huber band; the penalty then
    # acts like L1 with strength lam_h * delta -- a soft-threshold on b.
    rng = np.random.default_rng(11)
    n = 80
    x = rng.normal(size=n)
    # Strong correlation between x and -fpumz to push m_quad past delta.
    y_target = 5.0 * (x - x.mean()) + 0.05 * rng.normal(size=n)
    fpumz = -y_target
    rho = 1.5
    lam_h = 0.4
    delta = 0.5

    feat = _LinearFeature(
        name="x", regularization={"huber": {"coef": lam_h, "delta": delta}}
    )
    feat.initialize(x)
    feat.optimize(fpumz, rho=rho)

    x_centered = x - x.mean()
    xtx = float(x_centered.dot(x_centered))
    b = float(x_centered.dot(-fpumz))
    expected = _huber_closed_form(b, xtx, lam_h, delta, rho)
    assert abs(expected) > delta  # confirm we're in the linear region
    assert feat._m == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_huber_combined_with_l2_closed_form() -> None:
    rng = np.random.default_rng(13)
    n = 60
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)
    rho = 1.5
    lam_h = 0.5
    lam2 = 1.2
    delta = 0.4

    feat = _LinearFeature(
        name="x",
        regularization={
            "huber": {"coef": lam_h, "delta": delta},
            "l2": {"coef": lam2},
        },
    )
    feat.initialize(x)
    feat.optimize(fpumz, rho=rho)

    x_centered = x - x.mean()
    xtx = float(x_centered.dot(x_centered))
    b = float(x_centered.dot(-fpumz))
    expected = _huber_closed_form(b, xtx, lam_h, delta, rho, lam2=lam2)
    assert feat._m == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_huber_large_delta_matches_ridge() -> None:
    # As delta -> infinity Huber collapses to 0.5 * m^2, i.e. ridge with
    # strength lam_h / 2. Pick lam2 so the two configurations match exactly.
    rng = np.random.default_rng(17)
    n = 100
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)
    rho = 2.0
    lam_h = 1.4

    huber = _LinearFeature(
        name="x", regularization={"huber": {"coef": lam_h, "delta": 1e6}}
    )
    huber.initialize(x)
    huber.optimize(fpumz, rho=rho)

    ridge = _LinearFeature(name="x", regularization={"l2": {"coef": 0.5 * lam_h}})
    ridge.initialize(x)
    ridge.optimize(fpumz, rho=rho)

    assert huber._m == pytest.approx(ridge._m, rel=1e-10, abs=1e-12)


def test_huber_save_load_round_trip(tmp_path: Path) -> None:
    feat = _LinearFeature(
        name="x", regularization={"huber": {"coef": 0.7, "delta": 0.4}}
    )
    feat.initialize(
        np.arange(5.0),
        smoothing=2.0,
        save_flag=True,
        save_prefix=str(tmp_path / "model"),
    )
    feat._m = 1.25
    feat._b = -0.5
    feat._save()

    restored = _LinearFeature(load_from_file=feat._filename)
    assert restored._has_huber
    assert restored._coef_huber == pytest.approx(0.7)
    assert restored._lambda_huber == pytest.approx(1.4)
    assert restored._delta_huber == pytest.approx(0.4)
    assert restored._m == pytest.approx(1.25)


def test_huber_within_gam_bounds_outlier_pulled_slope() -> None:
    # End-to-end: compare ridge against huber-regularized fits when the
    # design includes a single high-leverage point. With strong shrinkage
    # the pure-ridge fit gets dragged toward 0; huber bounds the per-
    # coefficient influence and stays closer to the data-driven slope.
    rng = np.random.default_rng(2)
    n = 200
    x = rng.normal(size=n)
    y = 2.0 * (x - x.mean()) + 0.05 * rng.normal(size=n)
    X = pd.DataFrame({"x": x})

    coef = 5.0
    delta = 0.1
    huber = GAM(family="normal")
    huber.add_feature(
        name="x",
        type="linear",
        regularization={"huber": {"coef": coef, "delta": delta}},
    )
    huber.fit(X, y, max_its=80)

    ridge = GAM(family="normal")
    ridge.add_feature(name="x", type="linear", regularization={"l2": {"coef": coef}})
    ridge.fit(X, y, max_its=80)

    huber_slope = float(huber._features["x"]._m)  # type: ignore[attr-defined]
    ridge_slope = float(ridge._features["x"]._m)  # type: ignore[attr-defined]
    # Both shrink, but huber's bounded-influence form lets the slope stay
    # closer to the true value (2.0) than pure ridge with the same coef.
    assert abs(huber_slope - 2.0) < abs(ridge_slope - 2.0)
