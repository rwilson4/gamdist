"""Tests for the public GAM API: validation, predict shape, summary."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM


def test_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="not supported"):
        GAM(family="not_a_family")  # type: ignore[arg-type]


def test_no_family_raises() -> None:
    with pytest.raises(ValueError, match="Family not specified"):
        GAM()


def test_unknown_link_raises() -> None:
    with pytest.raises(ValueError, match="link not supported"):
        GAM(family="normal", link="not_a_link")  # type: ignore[arg-type]


def test_unknown_feature_type_raises() -> None:
    mdl = GAM(family="normal")
    with pytest.raises(ValueError, match="not supported"):
        mdl.add_feature(name="x", type="bogus")  # type: ignore[arg-type]


def test_predict_before_fit_raises() -> None:
    mdl = GAM(family="normal")
    mdl.add_feature(name="x", type="linear")
    with pytest.raises(AttributeError, match="not yet fit"):
        mdl.predict(pd.DataFrame({"x": [0.0, 1.0]}))


def test_save_without_name_raises() -> None:
    mdl = GAM(family="normal")
    mdl.add_feature(name="x", type="linear")
    X = pd.DataFrame({"x": np.arange(10.0)})
    y = X["x"].values
    with pytest.raises(ValueError, match="GAM with no name"):
        mdl.fit(X, y, save_flag=True)


def test_fit_predict_shape() -> None:
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = 2.0 * X["x"].values + rng.normal(size=n) * 0.1

    mdl = GAM(family="normal")
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=20)

    yhat = mdl.predict(X)
    assert yhat.shape == (n,)


def test_confidence_intervals_not_implemented() -> None:
    mdl = GAM(family="normal")
    mdl.add_feature(name="x", type="linear")
    X = pd.DataFrame({"x": np.arange(10.0)})
    y = X["x"].values
    mdl.fit(X, y, max_its=10)
    with pytest.raises(NotImplementedError):
        mdl.confidence_intervals(X)


def test_aicc_matches_formula() -> None:
    rng = np.random.default_rng(7)
    n = 200
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = 2.0 * X["x"].values + rng.normal(size=n) * 0.1
    mdl = GAM(family="normal")
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=20)

    p = mdl.dof() + (0.0 if mdl._known_dispersion else 1.0)
    expected = mdl.aic() + 2.0 * p * (p + 1) / (n - p - 1)
    assert mdl.aicc() == pytest.approx(expected)
    assert mdl.aicc() > mdl.aic()


def test_aicc_overparameterized_returns_inf() -> None:
    # n=3 with affine + linear-feature dof + estimated dispersion gives
    # p=3, so n - p - 1 = -1 <= 0 and AICc is undefined.
    X = pd.DataFrame({"x": np.array([0.0, 1.0, 2.0])})
    y = np.array([0.5, 1.5, 2.5])
    mdl = GAM(family="normal")
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=5)
    assert mdl.aicc() == float("inf")


def test_bic_matches_formula() -> None:
    rng = np.random.default_rng(11)
    n = 200
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = 2.0 * X["x"].values + rng.normal(size=n) * 0.1
    mdl = GAM(family="normal")
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=20)

    p = mdl.dof() + (0.0 if mdl._known_dispersion else 1.0)
    # BIC - AIC == (log(n) - 2) * p
    assert mdl.bic() - mdl.aic() == pytest.approx((np.log(n) - 2.0) * p)


def test_inconsistent_X_y_lengths_raises() -> None:
    mdl = GAM(family="normal")
    mdl.add_feature(name="x", type="linear")
    X = pd.DataFrame({"x": np.arange(10.0)})
    y = np.arange(5.0)
    with pytest.raises(ValueError, match="Inconsistent number of observations"):
        mdl.fit(X, y)


def test_fit_weights_wrong_length_raises() -> None:
    mdl = GAM(family="normal")
    mdl.add_feature(name="x", type="linear")
    X = pd.DataFrame({"x": np.arange(10.0)})
    y = np.arange(10.0)
    weights = np.ones(5)
    with pytest.raises(ValueError, match="weights has length"):
        mdl.fit(X, y, weights=weights)


def test_fit_covariate_class_sizes_wrong_length_raises() -> None:
    mdl = GAM(family="binomial")
    mdl.add_feature(name="x", type="linear")
    X = pd.DataFrame({"x": np.arange(10.0)})
    y = np.arange(10.0)
    ccs = np.full(7, 10.0)
    with pytest.raises(ValueError, match="covariate_class_sizes has length"):
        mdl.fit(X, y, covariate_class_sizes=ccs)


def test_summary_prints_features() -> None:
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = X["x"].values + rng.normal(size=n) * 0.1
    mdl = GAM(family="normal")
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=10)
    buf = io.StringIO()
    with redirect_stdout(buf):
        mdl.summary()
    text = buf.getvalue()
    assert "Model Statistics" in text
    assert "Features" in text
    assert "Feature x" in text


def test_canonical_links_default() -> None:
    assert GAM(family="normal")._link == "identity"
    assert GAM(family="binomial")._link == "logistic"
    assert GAM(family="poisson")._link == "log"
    assert GAM(family="gamma")._link == "reciprocal"
    assert GAM(family="inverse_gaussian")._link == "reciprocal_squared"


def test_exponential_collapses_to_gamma() -> None:
    mdl = GAM(family="exponential")
    assert mdl._family == "gamma"
    assert mdl._known_dispersion is True
    assert mdl._dispersion == 1.0


def _make_multi_feature_data(n: int = 200, seed: int = 0):
    """Three nontrivial features so n_jobs > 1 actually has work to dispatch."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.uniform(0.0, 1.0, size=n),
            "g": rng.choice(np.array(["a", "b", "c"]), size=n),
        }
    )
    y = (
        0.5 * X["x1"].values
        + 2.0 * X["x2"].values
        + np.where(X["g"].values == "a", 0.3, np.where(X["g"].values == "b", -0.1, 0.2))
        + rng.normal(size=n) * 0.1
    )
    return X, y


def _fit_with_n_jobs(X, y, n_jobs: int) -> GAM:
    mdl = GAM(family="normal")
    mdl.add_feature(name="x1", type="linear")
    mdl.add_feature(name="x2", type="spline")
    mdl.add_feature(name="g", type="categorical")
    mdl.fit(X, y, max_its=20, n_jobs=n_jobs)
    return mdl


def test_fit_n_jobs_2_matches_serial() -> None:
    X, y = _make_multi_feature_data()
    serial = _fit_with_n_jobs(X, y, n_jobs=1)
    parallel = _fit_with_n_jobs(X, y, n_jobs=2)
    # f_bar is the per-observation linear predictor accumulator -- the
    # primary state mutated by the loop. Serial and parallel paths
    # iterate self._features in insertion order on both, so accumulation
    # of f_new is bit-deterministic; the rest of ADMM is functional.
    np.testing.assert_allclose(serial.f_bar, parallel.f_bar, atol=1e-10)
    np.testing.assert_allclose(serial.predict(X), parallel.predict(X), atol=1e-10)


def test_fit_n_jobs_zero_raises() -> None:
    X, y = _make_multi_feature_data(n=20)
    mdl = GAM(family="normal")
    mdl.add_feature(name="x1", type="linear")
    with pytest.raises(ValueError, match="n_jobs must be"):
        mdl.fit(X, y, max_its=5, n_jobs=0)


def test_fit_n_jobs_minus_one_runs() -> None:
    X, y = _make_multi_feature_data(n=80)
    mdl = GAM(family="normal")
    mdl.add_feature(name="x1", type="linear")
    mdl.add_feature(name="g", type="categorical")
    mdl.fit(X, y, max_its=10, n_jobs=-1)
    assert mdl._fitted
    # Pool must be cleaned up when fit() returns.
    assert mdl._pool is None


def test_fit_pool_torn_down_after_fit() -> None:
    X, y = _make_multi_feature_data(n=80)
    mdl = GAM(family="normal")
    mdl.add_feature(name="x1", type="linear")
    mdl.add_feature(name="x2", type="spline")
    mdl.fit(X, y, max_its=10, n_jobs=2)
    assert mdl._pool is None  # ThreadPoolExecutor.shutdown() was called.
