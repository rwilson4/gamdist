"""Tests for the huber regularizer on _CategoricalFeature.

The penalty ``sum_c lambda_c * h_delta(q_c)`` is added to the cvxpy
objective. ``coef`` accepts a scalar (uniform across categories) or a
dict (per-category). ``delta`` is the band where the penalty is
quadratic (ridge-like); outside it grows linearly (L1-like), bounding
the per-coefficient influence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM
from gamdist.categorical_feature import _CategoricalFeature


def test_huber_no_coef_raises() -> None:
    with pytest.raises(ValueError, match="No coefficient specified for huber"):
        _CategoricalFeature(name="g", regularization={"huber": {"delta": 1.0}})


def test_huber_no_delta_raises() -> None:
    with pytest.raises(ValueError, match="No delta specified for huber"):
        _CategoricalFeature(name="g", regularization={"huber": {"coef": 1.0}})


@pytest.mark.parametrize("bad_delta", [0.0, -0.5])
def test_huber_non_positive_delta_raises(bad_delta: float) -> None:
    with pytest.raises(ValueError, match="huber delta must be positive"):
        _CategoricalFeature(
            name="g", regularization={"huber": {"coef": 1.0, "delta": bad_delta}}
        )


def test_huber_smoothing_scales_lambda_scalar_coef() -> None:
    feat = _CategoricalFeature(
        name="g", regularization={"huber": {"coef": 0.4, "delta": 1.0}}
    )
    feat.initialize(np.array(["a", "b", "a", "c"]), smoothing=2.5)
    np.testing.assert_allclose(feat._lambda_huber_vec, [1.0, 1.0, 1.0])
    assert feat._delta_huber == pytest.approx(1.0)


def test_huber_smoothing_scales_lambda_dict_coef() -> None:
    feat = _CategoricalFeature(
        name="g",
        regularization={"huber": {"coef": {"a": 1.0, "b": 0.5}, "delta": 0.4}},
    )
    feat.initialize(np.array(["a", "b", "c"]), smoothing=2.0)
    a = feat._lambda_huber_vec[feat._category_hash["a"]]
    b = feat._lambda_huber_vec[feat._category_hash["b"]]
    c = feat._lambda_huber_vec[feat._category_hash["c"]]
    assert a == pytest.approx(2.0)
    assert b == pytest.approx(1.0)
    assert c == pytest.approx(0.0)


def _fit_categorical(
    x: np.ndarray, y: np.ndarray, regularization: dict | None = None
) -> _CategoricalFeature:
    feat = _CategoricalFeature(name="g", regularization=regularization)
    feat.initialize(x)
    n = len(y)
    fpumz = -np.asarray(y, dtype=float) * n  # rough target so optimize sees the signal
    # Run a few rounds to settle since p starts at 0.
    for _ in range(20):
        feat.optimize(fpumz, rho=1.0)
    return feat


def test_huber_lambda_zero_matches_unpenalized() -> None:
    rng = np.random.default_rng(0)
    n = 200
    cats = rng.choice(["a", "b", "c"], size=n)
    y = rng.normal(size=n)

    plain = _fit_categorical(cats, y)
    zero = _fit_categorical(
        cats, y, regularization={"huber": {"coef": 0.0, "delta": 1.0}}
    )
    np.testing.assert_allclose(zero.p, plain.p, atol=1e-7)


def test_huber_huge_lambda_zeros_coefficients() -> None:
    rng = np.random.default_rng(0)
    n = 200
    cats = rng.choice(["a", "b", "c"], size=n)
    y = rng.normal(size=n)

    feat = _fit_categorical(
        cats, y, regularization={"huber": {"coef": 1e6, "delta": 1.0}}
    )
    np.testing.assert_allclose(feat.p, np.zeros_like(feat.p), atol=5e-3)


def test_huber_large_delta_matches_ridge() -> None:
    # As delta -> infinity Huber collapses to 0.5 * q^2 elementwise, i.e.
    # ridge with strength lam_h / 2.
    rng = np.random.default_rng(1)
    n = 300
    cats = rng.choice(["a", "b", "c", "d"], size=n)
    y = rng.normal(size=n)

    lam = 1.0
    huber_feat = _fit_categorical(
        cats, y, regularization={"huber": {"coef": lam, "delta": 1e4}}
    )
    ridge_feat = _fit_categorical(cats, y, regularization={"l2": {"coef": 0.5 * lam}})
    np.testing.assert_allclose(huber_feat.p, ridge_feat.p, atol=1e-3)


def test_huber_save_load_round_trip(tmp_path: Path) -> None:
    feat = _CategoricalFeature(
        name="g", regularization={"huber": {"coef": 0.7, "delta": 0.4}}
    )
    feat.initialize(
        np.array(["a", "b", "a", "c"]),
        smoothing=2.0,
        save_flag=True,
        save_prefix=str(tmp_path / "model"),
    )
    feat.p = np.array([0.5, -0.5, 0.0])
    feat._save()

    restored = _CategoricalFeature(load_from_file=feat._filename)
    assert restored._has_huber
    assert restored._coef_huber == pytest.approx(0.7)
    assert restored._delta_huber == pytest.approx(0.4)
    np.testing.assert_allclose(restored._lambda_huber_vec, [1.4, 1.4, 1.4])
    np.testing.assert_allclose(restored.p, [0.5, -0.5, 0.0])


def test_huber_within_gam_shrinks_extreme_categories() -> None:
    # Three categories: one with a huge true effect, two near zero. With
    # huber regularization, the extreme-category coefficient saturates at
    # the linear region (slope lam * delta) rather than uniformly
    # contracting toward zero like pure ridge with the same lam.
    rng = np.random.default_rng(3)
    n_per = 200
    n = 3 * n_per
    cats = np.array(["a"] * n_per + ["b"] * n_per + ["c"] * n_per)
    rng.shuffle(cats)
    truth = {"a": 5.0, "b": 0.05, "c": -0.05}
    y = np.array([truth[c] for c in cats]) + 0.05 * rng.normal(size=n)
    X = pd.DataFrame({"g": cats})

    lam = 1.0
    delta = 0.5
    huber = GAM(family="normal")
    huber.add_feature(
        name="g",
        type="categorical",
        regularization={"huber": {"coef": lam, "delta": delta}},
    )
    huber.fit(X, y, max_its=100)

    ridge = GAM(family="normal")
    ridge.add_feature(
        name="g", type="categorical", regularization={"l2": {"coef": lam}}
    )
    ridge.fit(X, y, max_its=100)

    h = huber._features["g"].p  # type: ignore[attr-defined]
    r = ridge._features["g"].p  # type: ignore[attr-defined]
    feat = huber._features["g"]
    a_idx = feat._category_hash["a"]  # type: ignore[attr-defined]
    # The extreme coefficient gets less aggressive shrinkage from huber
    # than from pure ridge with the same lambda.
    assert abs(h[a_idx]) > abs(r[a_idx])
