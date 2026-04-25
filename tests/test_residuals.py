"""Tests for GAM.residuals() and GAM.plot_residuals()."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM


@pytest.fixture
def normal_fit() -> GAM:
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = X["x"].values + rng.normal(size=n) * 0.1
    mdl = GAM(family="normal")
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=20)
    return mdl


def test_residuals_response_is_y_minus_mu(normal_fit: GAM) -> None:
    y = normal_fit._y
    mu = normal_fit._eval_inv_link(normal_fit._num_features * normal_fit.f_bar)
    np.testing.assert_allclose(normal_fit.residuals("response"), y - mu)


def test_residuals_deviance_normal_equals_response(normal_fit: GAM) -> None:
    # For normal family, deviance residual is sign(y - mu) * |y - mu| = y - mu.
    np.testing.assert_allclose(
        normal_fit.residuals("deviance"),
        normal_fit.residuals("response"),
        atol=1e-12,
    )


def test_residuals_pearson_normal_scaled_by_sqrt_phi(normal_fit: GAM) -> None:
    phi = normal_fit.dispersion()
    expected = normal_fit.residuals("response") / np.sqrt(phi)
    np.testing.assert_allclose(normal_fit.residuals("pearson"), expected)


def test_deviance_residual_squared_sum_matches_deviance_normal(normal_fit: GAM) -> None:
    # For normal family, sum(d_i) == deviance() exactly.
    d_residuals = normal_fit.residuals("deviance")
    assert float(np.sum(d_residuals**2)) == pytest.approx(normal_fit.deviance())


def test_residuals_invalid_kind_raises(normal_fit: GAM) -> None:
    with pytest.raises(ValueError, match="Unknown residual kind"):
        normal_fit.residuals("bogus")  # type: ignore[arg-type]


def test_residuals_before_fit_raises() -> None:
    mdl = GAM(family="normal")
    mdl.add_feature(name="x", type="linear")
    with pytest.raises(AttributeError, match="not yet fit"):
        mdl.residuals()


def test_residuals_poisson_finite() -> None:
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = rng.poisson(np.exp(0.4 * X["x"].values + 0.5), size=n).astype(float)
    mdl = GAM(family="poisson")
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=30)
    for kind in ("response", "pearson", "deviance"):
        r = mdl.residuals(kind)  # type: ignore[arg-type]
        assert r.shape == (n,)
        assert np.all(np.isfinite(r))


def test_residuals_binomial_with_ccs() -> None:
    rng = np.random.default_rng(2)
    n = 50
    X = pd.DataFrame({"x": rng.normal(size=n)})
    ccs = np.full(n, 10.0)
    p = 1.0 / (1.0 + np.exp(-X["x"].values))
    y = rng.binomial(10, p).astype(float)
    mdl = GAM(family="binomial")
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, covariate_class_sizes=ccs, max_its=30)
    for kind in ("response", "pearson", "deviance"):
        r = mdl.residuals(kind)  # type: ignore[arg-type]
        assert r.shape == (n,)
        assert np.all(np.isfinite(r))


def test_plot_residuals_returns_figure(normal_fit: GAM) -> None:
    fig = normal_fit.plot_residuals(kind="deviance")
    # 1x2 layout: residuals-vs-fitted, QQ
    assert len(fig.axes) == 2
    titles = {ax.get_title() for ax in fig.axes}
    assert "Residuals vs Fitted" in titles
    assert "Normal Q-Q" in titles
