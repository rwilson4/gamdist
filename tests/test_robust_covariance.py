"""Sandwich (Huber-White) covariance estimator vs statsmodels.

The sandwich estimator is a post-fit linear-algebra construction off
the fitted ``mu`` -- it does not retrace the optimization path. So as
long as gamdist's converged ``mu`` matches statsmodels' MLE on the
same model, ``robust_covariance(cov_type='HC0')`` should match
statsmodels' ``GLM(...).fit(cov_type='HC0').cov_params()`` exactly,
modulo the optimization tolerance that separates the two fits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM

sm = pytest.importorskip("statsmodels.api")


def _design_for_statsmodels(X: pd.DataFrame, mdl: GAM) -> np.ndarray:
    """Mirror GAM._design_matrix in the statsmodels parametrization."""
    D, _ = mdl._design_matrix(X)
    return D


def _gam_fit(family, link=None, **fit_kwargs):
    rng = np.random.default_rng(7)
    n = 600
    x1 = rng.normal(size=n)
    x2 = rng.uniform(-1.0, 2.0, size=n)
    cat = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])
    cat_eff = np.where(cat == "a", 0.0, np.where(cat == "b", 0.4, -0.3))
    eta = 0.5 + 0.7 * x1 - 0.3 * x2 + cat_eff

    if family == "normal":
        mu = eta
        y = mu + rng.normal(scale=0.5, size=n)
    elif family == "binomial":
        mu = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, mu).astype(float)
    elif family == "poisson":
        mu = np.exp(eta * 0.3)  # keep counts modest
        y = rng.poisson(mu).astype(float)
    else:
        raise ValueError(family)

    X = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat})

    mdl = GAM(family=family, link=link)
    mdl.add_feature(name="x1", type="linear")
    mdl.add_feature(name="x2", type="linear")
    mdl.add_feature(name="cat", type="categorical")
    mdl.fit(X, y, max_its=200, **fit_kwargs)
    return mdl, X, y


def _statsmodels_glm(family, X, y, mdl):
    if family == "normal":
        sm_family = sm.families.Gaussian()
    elif family == "binomial":
        sm_family = sm.families.Binomial()
    elif family == "poisson":
        sm_family = sm.families.Poisson()
    else:
        raise ValueError(family)
    D = _design_for_statsmodels(X, mdl)
    return sm.GLM(y, D, family=sm_family).fit(cov_type="HC0")


def _eta_from_mu(family: str, mu: np.ndarray) -> np.ndarray:
    if family == "normal":
        return mu
    if family == "binomial":
        eps = np.finfo(float).eps
        mc = np.clip(mu, eps, 1.0 - eps)
        return np.log(mc / (1.0 - mc))
    if family == "poisson":
        return np.log(mu)
    raise ValueError(family)


@pytest.mark.parametrize("family", ["normal", "binomial", "poisson"])
def test_robust_covariance_formula_matches_statsmodels_at_same_mu(
    family: str,
) -> None:
    """Pin the sandwich formula independent of the optimizer.

    gamdist's ADMM and statsmodels' IRLS converge to slightly different
    points (eps_abs=1e-3 vs. machine precision), so a head-to-head
    comparison of ``cov_params()`` mixes formula error with optimizer
    error. Inject statsmodels' converged ``mu`` into gamdist and
    verify the sandwich matches to numerical precision -- this is the
    sharp test of correctness.
    """
    mdl, X, y = _gam_fit(family)
    sm_res = _statsmodels_glm(family, X, y, mdl)

    # Replace gamdist's converged eta with statsmodels' converged eta.
    # robust_covariance reads mu via self._eval_inv_link(N * self.f_bar),
    # so writing f_bar = eta_sm / N gives mu = mu_sm exactly.
    D = _design_for_statsmodels(X, mdl)
    eta_sm = sm_res.predict(D, which="linear")
    mdl.f_bar = np.asarray(eta_sm, dtype=float) / mdl._num_features

    V_gam = mdl.robust_covariance(cov_type="HC0")
    V_sm = np.asarray(sm_res.cov_params())
    assert V_gam.shape == V_sm.shape
    np.testing.assert_allclose(V_gam, V_sm, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("family", ["normal", "binomial", "poisson"])
def test_robust_covariance_end_to_end_close_to_statsmodels(family: str) -> None:
    """End-to-end: fit with gamdist, compute sandwich, compare.

    ADMM converges to ~1e-3 in primal/dual residuals, so the
    fitted ``mu`` differs from statsmodels' MLE in roughly that
    range, and the sandwich differs accordingly. We check looser
    relative agreement; the tight pin on the formula is in the
    test above.
    """
    mdl, X, y = _gam_fit(family)
    sm_res = _statsmodels_glm(family, X, y, mdl)
    V_gam = mdl.robust_covariance(cov_type="HC0")
    V_sm = np.asarray(sm_res.cov_params())
    np.testing.assert_allclose(V_gam, V_sm, rtol=5e-2, atol=5e-3)


def test_robust_covariance_hc1_scales_hc0() -> None:
    mdl, _, _ = _gam_fit("normal")
    V0 = mdl.robust_covariance(cov_type="HC0")
    V1 = mdl.robust_covariance(cov_type="HC1")
    n = mdl._num_obs
    p = V0.shape[0]
    np.testing.assert_allclose(V1, V0 * (n / (n - p)), rtol=1e-12)


def test_robust_covariance_design_matrix_layout() -> None:
    mdl, X, _ = _gam_fit("normal")
    D, names = mdl._design_matrix()
    n = mdl._num_obs
    # 1 intercept + 2 linear + (3 - 1) categorical = 5 columns
    assert D.shape == (n, 5)
    assert names[0] == "Intercept"
    assert "x1" in names and "x2" in names
    assert any("[T." in nm for nm in names)
    # Intercept column is all ones.
    np.testing.assert_array_equal(D[:, 0], np.ones(n))


def test_robust_covariance_unfitted_raises() -> None:
    mdl = GAM(family="normal")
    mdl.add_feature(name="x1", type="linear")
    with pytest.raises(AttributeError):
        mdl.robust_covariance()


def test_robust_covariance_rejects_unsupported_cov_type() -> None:
    mdl, _, _ = _gam_fit("normal")
    with pytest.raises(NotImplementedError):
        mdl.robust_covariance(cov_type="HC3")


def test_robust_covariance_rejects_spline_features() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0, 1, size=n)
    y = np.sin(2.0 * np.pi * x) + rng.normal(scale=0.1, size=n)
    X = pd.DataFrame({"x": x})
    mdl = GAM(family="normal")
    mdl.add_feature(name="x", type="spline")
    mdl.fit(X, y, max_its=80)
    with pytest.raises(NotImplementedError):
        mdl.robust_covariance()


def test_robust_covariance_rejects_quantile_family() -> None:
    rng = np.random.default_rng(1)
    n = 300
    x = rng.normal(size=n)
    y = 1.0 + 2.0 * x + rng.normal(size=n)
    X = pd.DataFrame({"x": x})
    mdl = GAM(family="quantile", tau=0.5)
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=80)
    with pytest.raises(NotImplementedError):
        mdl.robust_covariance()
