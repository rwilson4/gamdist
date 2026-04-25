"""Unit tests for the per-(family, link) proximal operators."""

from __future__ import annotations

import numpy as np

from gamdist import proximal_operators as po


def test_normal_identity_no_weights() -> None:
    v = np.array([1.0, 2.0, 3.0])
    y = np.array([0.5, 1.5, 2.5])
    mu = 2.0
    out = po._prox_normal_identity(v, mu, y)
    expected = (y + mu * v) / (1.0 + mu)
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_normal_identity_with_weights() -> None:
    v = np.array([1.0, 2.0, 3.0])
    y = np.array([0.5, 1.5, 2.5])
    w = np.array([0.5, 1.0, 2.0])
    mu = 2.0
    out = po._prox_normal_identity(v, mu, y, w=w)
    expected = (w * y + mu * v) / (w + mu)
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_gamma_reciprocal_no_weights() -> None:
    v = np.array([0.5, 1.0])
    y = np.array([1.0, 2.0])
    mu = 1.0
    out = po._prox_gamma_reciprocal(v, mu, y)
    diff = mu * v - y
    expected = (0.5 / mu) * diff + np.sqrt(diff * diff + 4.0 * mu)
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_binomial_logit_scalar_converges() -> None:
    out = po._prox_binomial_logit_scalar([0.0, 1.0, 0.5, 1.0])
    assert isinstance(out, float)
    assert np.isfinite(out)


def test_binomial_logit_array() -> None:
    rng = np.random.default_rng(0)
    n = 8
    v = rng.normal(size=n)
    y = rng.integers(0, 2, size=n).astype(float)
    out = po._prox_binomial_logit(v, mu=1.5, y=y)
    assert out.shape == (n,)
    assert np.all(np.isfinite(out))


def test_binomial_logit_with_ccs_and_weights() -> None:
    rng = np.random.default_rng(1)
    n = 6
    v = rng.normal(size=n)
    y = rng.integers(0, 5, size=n).astype(float)
    ccs = np.full(n, 5.0)
    w = np.ones(n)
    out = po._prox_binomial_logit(v, mu=1.0, y=y, ccs=ccs, w=w)
    assert out.shape == (n,)
    assert np.all(np.isfinite(out))


def test_poisson_log_scalar_converges() -> None:
    out = po._prox_poisson_log_scalar([0.0, 1.0, 2.0])
    assert isinstance(out, float)
    assert np.isfinite(out)


def test_inv_gaussian_reciprocal_squared_scalar_positive() -> None:
    out = po._prox_inv_gaussian_reciprocal_squared_scalar([0.5, 1.0, 1.0])
    assert out > 0


def test_binomial_logit_scalar_with_weight() -> None:
    out = po._prox_binomial_logit_scalar([0.0, 1.0, 0.5, 1.0, 2.0])
    assert isinstance(out, float)
    assert np.isfinite(out)


def test_binomial_logit_scalar_handles_extreme_v() -> None:
    # v=50 used to drive Newton into np.exp() overflow on the first
    # iteration. The hardened version stays finite via expit() and
    # backtracking.
    out = po._prox_binomial_logit_scalar([50.0, 0.1, 0.0, 1.0])
    assert np.isfinite(out)


def test_binomial_logit_scalar_matches_brute_force_optimum() -> None:
    # Spot-check the Newton solution against scipy.optimize.minimize_scalar
    # on the same objective.
    v, mu, y, m = 0.7, 1.5, 0.5, 1.0

    def obj(z: float) -> float:
        return m * float(np.logaddexp(0.0, z)) - y * z + 0.5 * mu * (z - v) ** 2

    from scipy.optimize import minimize_scalar

    expected = float(minimize_scalar(obj).x)
    out = po._prox_binomial_logit_scalar([v, mu, y, m])
    assert abs(out - expected) < 1e-3


def test_poisson_log_scalar_handles_extreme_v() -> None:
    # Large positive v previously could cause exp(x) to overflow during
    # Newton; the cap + damping keeps the result finite.
    out = po._prox_poisson_log_scalar([20.0, 0.1, 0.0])
    assert np.isfinite(out)
