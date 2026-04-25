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


def test_anscombe_normal_equals_pearson(normal_fit: GAM) -> None:
    # For normal family V(mu)=1 and A(t)=t, so Anscombe collapses to
    # the Pearson residual.
    np.testing.assert_allclose(
        normal_fit.residuals("anscombe"),
        normal_fit.residuals("pearson"),
        atol=1e-12,
    )


def test_anscombe_poisson_matches_formula() -> None:
    rng = np.random.default_rng(7)
    n = 200
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = rng.poisson(np.exp(0.4 * X["x"].values + 0.5), size=n).astype(float)
    mdl = GAM(family="poisson")
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=30)

    mu = mdl._eval_inv_link(mdl._num_features * mdl.f_bar)
    expected = 1.5 * (y ** (2 / 3) - mu ** (2 / 3)) / mu ** (1 / 6) / np.sqrt(
        mdl.dispersion()
    )
    np.testing.assert_allclose(mdl.residuals("anscombe"), expected, rtol=1e-10)


def test_anscombe_gamma_matches_formula() -> None:
    # Use known dispersion to skip the (numerically finicky)
    # _gamma_dispersion Newton estimator for the test.
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({"x": rng.uniform(0.1, 0.5, size=n)})
    y = rng.gamma(shape=4.0, scale=0.5, size=n)
    mdl = GAM(family="gamma", dispersion=1.0)
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=30)

    mu = mdl._eval_inv_link(mdl._num_features * mdl.f_bar)
    expected = 3.0 * (y ** (1 / 3) - mu ** (1 / 3)) / mu ** (1 / 3)
    np.testing.assert_allclose(mdl.residuals("anscombe"), expected, rtol=1e-10)


def test_anscombe_binomial_with_ccs_finite() -> None:
    rng = np.random.default_rng(2)
    n = 50
    X = pd.DataFrame({"x": rng.normal(size=n)})
    ccs = np.full(n, 10.0)
    p = 1.0 / (1.0 + np.exp(-X["x"].values))
    y = rng.binomial(10, p).astype(float)
    mdl = GAM(family="binomial")
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, covariate_class_sizes=ccs, max_its=30)
    r = mdl.residuals("anscombe")
    assert r.shape == (n,)
    assert np.all(np.isfinite(r))


def test_anscombe_inverse_gaussian_matches_formula() -> None:
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"x": rng.uniform(0.1, 1.0, size=n)})
    y = rng.uniform(0.5, 1.5, size=n)
    mdl = GAM(family="inverse_gaussian")
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=20)

    mu = mdl._eval_inv_link(mdl._num_features * mdl.f_bar)
    expected = (np.log(y) - np.log(mu)) / np.sqrt(mu * mdl.dispersion())
    np.testing.assert_allclose(mdl.residuals("anscombe"), expected, rtol=1e-10)


def test_plot_residuals_accepts_anscombe(normal_fit: GAM) -> None:
    fig = normal_fit.plot_residuals(kind="anscombe")
    assert len(fig.axes) == 2


def test_plot_residuals_vs_predictor_smoke(normal_fit: GAM) -> None:
    rng = np.random.default_rng(0)
    predictor = rng.normal(size=normal_fit._num_obs)
    fig = normal_fit.plot_residuals_vs_predictor(predictor, name="x")
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_title() == "Residuals vs x"
    assert ax.get_xlabel() == "x"


def test_plot_residuals_vs_predictor_default_label(normal_fit: GAM) -> None:
    predictor = np.linspace(0.0, 1.0, normal_fit._num_obs)
    fig = normal_fit.plot_residuals_vs_predictor(predictor)
    ax = fig.axes[0]
    # No name supplied -> generic label.
    assert ax.get_xlabel() == "predictor"
    assert ax.get_title() == "Residuals vs predictor"


def test_plot_residuals_vs_predictor_categorical(normal_fit: GAM) -> None:
    rng = np.random.default_rng(0)
    predictor = rng.choice(np.array(["a", "b", "c"]), size=normal_fit._num_obs)
    fig = normal_fit.plot_residuals_vs_predictor(predictor, name="group")
    # matplotlib auto-creates a categorical x-axis; just smoke-test that
    # the figure exists and the title carries the name.
    assert fig.axes[0].get_title() == "Residuals vs group"


def test_plot_residuals_vs_predictor_shape_mismatch_raises(normal_fit: GAM) -> None:
    bad = np.zeros(normal_fit._num_obs + 1)
    with pytest.raises(ValueError, match="expected"):
        normal_fit.plot_residuals_vs_predictor(bad)


def test_plot_residuals_vs_predictor_uses_chosen_kind(normal_fit: GAM) -> None:
    predictor = np.arange(normal_fit._num_obs, dtype=float)
    fig = normal_fit.plot_residuals_vs_predictor(predictor, kind="pearson")
    assert fig.axes[0].get_ylabel() == "Pearson residual"
