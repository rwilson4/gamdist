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
