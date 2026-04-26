"""Quasi-likelihood families: ``quasi_poisson`` and ``quasi_binomial``.

These families share the score function (and therefore the per-feature
prox, deviance, and variance function) with their full-likelihood
cousins; only the dispersion estimator differs. The tests here pin
both halves of that contract: point estimates match, dispersion is
estimated rather than fixed at one, and overdispersed data drives the
estimate above one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM


def _negative_binomial_counts(
    rng: np.random.Generator, mu: np.ndarray, theta: float
) -> np.ndarray:
    """Draw NB(mu, theta) counts via the Poisson-Gamma mixture.

    ``Var(Y) = mu + mu^2 / theta``, so smaller ``theta`` is more
    overdispersed. ``theta -> infinity`` recovers the Poisson.
    """
    shape = theta
    scale = mu / theta
    rate = rng.gamma(shape=shape, scale=scale)
    return rng.poisson(rate)


def _beta_binomial_counts(
    rng: np.random.Generator, m: np.ndarray, mu: np.ndarray, rho: float
) -> np.ndarray:
    """Draw beta-binomial counts: variance ``m mu (1-mu) (1 + (m-1) rho)``.

    ``rho in [0, 1)`` is the intra-cluster correlation; ``rho = 0`` is
    the exact binomial. ``alpha = mu (1/rho - 1)``,
    ``beta = (1 - mu) (1/rho - 1)``.
    """
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0, 1) for the beta-binomial draw.")
    inv_rho_minus_one = (1.0 - rho) / rho
    alpha = mu * inv_rho_minus_one
    beta = (1.0 - mu) * inv_rho_minus_one
    p = rng.beta(alpha, beta)
    return rng.binomial(m, p)


@pytest.fixture
def overdispersed_poisson_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(11)
    n = 400
    x = rng.normal(size=n)
    eta = 0.5 + 0.6 * x
    mu = np.exp(eta)
    y = _negative_binomial_counts(rng, mu, theta=2.0).astype(float)
    return pd.DataFrame({"x": x}), y


@pytest.fixture
def overdispersed_binomial_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(13)
    n = 200
    x = rng.normal(size=n)
    eta = -0.2 + 0.8 * x
    mu = 1.0 / (1.0 + np.exp(-eta))
    m = np.full(n, 50, dtype=float)
    y = _beta_binomial_counts(rng, m.astype(int), mu, rho=0.15).astype(float)
    return pd.DataFrame({"x": x}), y, m


def _fit_linear(family: str, X: pd.DataFrame, y: np.ndarray, **kwargs) -> GAM:
    mdl = GAM(family=family, **kwargs)
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, max_its=100)
    return mdl


def _fit_binomial_with_classes(
    family: str, X: pd.DataFrame, y: np.ndarray, m: np.ndarray
) -> GAM:
    mdl = GAM(family=family)
    mdl.add_feature(name="x", type="linear")
    mdl.fit(X, y, covariate_class_sizes=m, max_its=100)
    return mdl


def test_quasi_poisson_canonical_link_is_log() -> None:
    mdl = GAM(family="quasi_poisson")
    assert mdl._link == "log"
    assert mdl._base_family == "poisson"


def test_quasi_binomial_canonical_link_is_logistic() -> None:
    mdl = GAM(family="quasi_binomial")
    assert mdl._link == "logistic"
    assert mdl._base_family == "binomial"


def test_quasi_poisson_matches_poisson_point_estimates(
    overdispersed_poisson_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = overdispersed_poisson_data
    mdl_poisson = _fit_linear("poisson", X, y)
    mdl_quasi = _fit_linear("quasi_poisson", X, y)
    # Same prox, same data, same starting offset (log mean) -> ADMM
    # lands on the same fitted mean to within optimization tolerance.
    np.testing.assert_allclose(
        mdl_quasi.predict(X), mdl_poisson.predict(X), rtol=1e-6, atol=1e-6
    )


def test_quasi_poisson_estimates_dispersion_above_one(
    overdispersed_poisson_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = overdispersed_poisson_data
    mdl_quasi = _fit_linear("quasi_poisson", X, y)
    mdl_poisson = _fit_linear("poisson", X, y)
    phi = mdl_quasi.dispersion()
    # NB with theta=2 is markedly overdispersed; phi should be well
    # above 1 (typically several).
    assert phi > 1.5
    # Plain Poisson keeps phi pinned at 1.
    assert mdl_poisson.dispersion() == 1.0


def test_quasi_poisson_dispersion_matches_pearson_formula(
    overdispersed_poisson_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = overdispersed_poisson_data
    mdl = _fit_linear("quasi_poisson", X, y)
    mu = mdl.predict(X)
    expected = float(np.sum((y - mu) ** 2 / mu) / (len(y) - mdl.dof()))
    assert mdl.dispersion() == pytest.approx(expected, rel=1e-10)


def test_quasi_binomial_matches_binomial_point_estimates(
    overdispersed_binomial_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    X, y, m = overdispersed_binomial_data
    mdl_binomial = _fit_binomial_with_classes("binomial", X, y, m)
    mdl_quasi = _fit_binomial_with_classes("quasi_binomial", X, y, m)
    np.testing.assert_allclose(
        mdl_quasi.predict(X), mdl_binomial.predict(X), rtol=1e-6, atol=1e-6
    )


def test_quasi_binomial_estimates_dispersion_above_one(
    overdispersed_binomial_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    X, y, m = overdispersed_binomial_data
    mdl_quasi = _fit_binomial_with_classes("quasi_binomial", X, y, m)
    mdl_binomial = _fit_binomial_with_classes("binomial", X, y, m)
    phi = mdl_quasi.dispersion()
    # Beta-binomial(rho=0.15, m=50) gives a dispersion factor of
    # roughly 1 + (m-1) rho ~= 8.4; the estimate should land
    # comfortably above 1.
    assert phi > 2.0
    assert mdl_binomial.dispersion() == 1.0


def test_quasi_binomial_dispersion_matches_pearson_formula(
    overdispersed_binomial_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    X, y, m = overdispersed_binomial_data
    mdl = _fit_binomial_with_classes("quasi_binomial", X, y, m)
    mu = mdl.predict(X)
    eps = np.finfo(float).eps
    mu_c = np.clip(mu, eps, 1.0 - eps)
    expected = float(
        np.sum((y - m * mu_c) ** 2 / (m * mu_c * (1.0 - mu_c))) / (len(y) - mdl.dof())
    )
    assert mdl.dispersion() == pytest.approx(expected, rel=1e-10)


def test_quasi_binomial_works_on_bernoulli() -> None:
    """Quasi-binomial should run on raw 0/1 data even though
    overdispersion is not meaningfully estimable from Bernoulli alone."""
    rng = np.random.default_rng(7)
    n = 200
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.2 + 0.5 * x)))
    y = (rng.uniform(size=n) < p).astype(float)
    mdl_quasi = _fit_linear("quasi_binomial", pd.DataFrame({"x": x}), y)
    mdl_binomial = _fit_linear("binomial", pd.DataFrame({"x": x}), y)
    np.testing.assert_allclose(
        mdl_quasi.predict(pd.DataFrame({"x": x})),
        mdl_binomial.predict(pd.DataFrame({"x": x})),
        rtol=1e-6,
        atol=1e-6,
    )
    # For Bernoulli the Pearson chi-square / (n - p) is finite even
    # though the underlying notion of "overdispersion" is degenerate.
    assert np.isfinite(mdl_quasi.dispersion())


def test_quasi_poisson_user_supplied_dispersion_respected() -> None:
    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(0.3 + 0.4 * x), size=n).astype(float)
    mdl = GAM(family="quasi_poisson", dispersion=2.5)
    mdl.add_feature(name="x", type="linear")
    mdl.fit(pd.DataFrame({"x": x}), y, max_its=40)
    assert mdl.dispersion() == pytest.approx(2.5)


def test_quasi_poisson_rejects_non_canonical_link() -> None:
    with pytest.raises(ValueError, match="not a supported"):
        GAM(family="quasi_poisson", link="identity")


def test_quasi_binomial_rejects_non_canonical_link() -> None:
    with pytest.raises(ValueError, match="not a supported"):
        GAM(family="quasi_binomial", link="probit")


def test_quasi_poisson_robust_covariance_runs(
    overdispersed_poisson_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = overdispersed_poisson_data
    mdl = _fit_linear("quasi_poisson", X, y)
    V = mdl.robust_covariance()
    # The sandwich is dispersion-free, so it agrees with the Poisson
    # sandwich at the same fitted mu (point estimates already pinned
    # equal above).
    mdl_poisson = _fit_linear("poisson", X, y)
    V_p = mdl_poisson.robust_covariance()
    np.testing.assert_allclose(V, V_p, rtol=1e-4, atol=1e-6)


def test_quasi_poisson_save_load_round_trip(tmp_path) -> None:
    rng = np.random.default_rng(1)
    n = 80
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(0.2 + 0.3 * x), size=n).astype(float)
    name = str(tmp_path / "qp")
    mdl = GAM(family="quasi_poisson", name=name)
    mdl.add_feature(name="x", type="linear")
    mdl.fit(pd.DataFrame({"x": x}), y, save_flag=True, max_its=20)
    phi = mdl.dispersion()

    mdl2 = GAM(load_from_file=f"{name}_model.pckl")
    assert mdl2._family == "quasi_poisson"
    assert mdl2._base_family == "poisson"
    np.testing.assert_allclose(mdl2.dispersion(), phi)
