"""Cover the family/link branches of GAM._optimize and the metric methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM


@pytest.fixture
def small_normal_fit() -> GAM:
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = X["x"].values + rng.normal(size=n) * 0.05
    mdl = GAM(family="normal")
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=20)
    return mdl


def test_aic_finite(small_normal_fit: GAM) -> None:
    assert np.isfinite(small_normal_fit.aic())


def test_dispersion_finite(small_normal_fit: GAM) -> None:
    assert np.isfinite(small_normal_fit.dispersion())


def test_gcv_finite(small_normal_fit: GAM) -> None:
    assert np.isfinite(small_normal_fit.gcv())


def test_ubre_finite() -> None:
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = (rng.binomial(1, p=0.5, size=n)).astype(float)
    mdl = GAM(family="binomial")
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=20)
    assert np.isfinite(mdl.ubre())
    assert mdl.dispersion() == 1.0


def test_poisson_log_fits() -> None:
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = rng.poisson(np.exp(0.5 * X["x"].values + 0.5), size=n).astype(float)
    mdl = GAM(family="poisson")
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=40)
    assert np.isfinite(mdl.deviance())
    assert mdl.dispersion() == 1.0


def test_gamma_reciprocal_fits() -> None:
    # Gamma + reciprocal link can produce non-positive mu if the linear
    # predictor crosses zero, which leaves log(y/mu) undefined. Use a
    # near-constant target so mu stays well away from zero.
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({"x": rng.uniform(0.1, 0.5, size=n)})
    y = rng.gamma(shape=4.0, scale=0.5, size=n)
    mdl = GAM(family="gamma")
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=40)
    yhat = mdl.predict(X)
    assert np.all(yhat > 0)


def test_inverse_gaussian_fits() -> None:
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"x": rng.uniform(0.1, 1.0, size=n)})
    y = rng.uniform(0.5, 1.5, size=n)
    mdl = GAM(family="inverse_gaussian")
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=20)
    assert np.isfinite(mdl.deviance())


def test_known_dispersion_returned_directly() -> None:
    mdl = GAM(family="normal", dispersion=0.7)
    rng = np.random.default_rng(0)
    n = 50
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = X["x"].values + rng.normal(size=n) * 0.1
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=10)
    assert mdl.dispersion() == pytest.approx(0.7)


def test_dof_aggregates_features() -> None:
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "g": rng.choice(["a", "b", "c"], size=n),
        }
    )
    y = X["x"].values + np.where(
        X["g"].values == "a", 0.3, np.where(X["g"].values == "b", -0.1, 0.0)
    )
    mdl = GAM(family="normal")
    mdl.add_feature(name="x", type="linear")
    mdl.add_feature(name="g", type="categorical")
    mdl.fit(X, y, max_its=20)
    # 1 (intercept) + 1 (linear) + (3 categories - 1) = 4
    assert mdl.dof() == pytest.approx(4.0)
