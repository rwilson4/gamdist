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


def _pinball(r: float, tau: float) -> float:
    return float(max(tau * r, (tau - 1.0) * r))


def test_quantile_identity_matches_brute_force_optimum() -> None:
    # Prox solves min_x  rho_tau(y - x) + (mu/2) (x - v)^2 in closed
    # form; spot-check it against minimize_scalar on the same scalar
    # objective across (v, y) sign combinations and tau values.
    from scipy.optimize import minimize_scalar

    rng = np.random.default_rng(0)
    mu = 1.7
    for tau in (0.1, 0.5, 0.75):
        v = rng.normal(size=20)
        y = rng.normal(size=20)
        out = po._prox_quantile_identity(v, mu, y, tau=tau)
        for i in range(len(v)):

            def obj(
                x: float, _v: float = v[i], _y: float = y[i], _tau: float = tau
            ) -> float:
                return _pinball(_y - x, _tau) + 0.5 * mu * (x - _v) ** 2

            expected = float(minimize_scalar(obj).x)
            assert abs(out[i] - expected) < 1e-5


def test_quantile_identity_thresholds_and_passthrough() -> None:
    # Targets sandwiched between the upper and lower thresholds pass
    # through unchanged; outside, the prox saturates to the relevant
    # shifted-soft-threshold corner.
    mu = 2.0
    tau = 0.3
    v = np.array([0.0, 0.0, 0.0])
    upper = tau / mu  # +0.15
    lower = -(1.0 - tau) / mu  # -0.35
    y = np.array([upper - 0.05, upper + 0.5, lower - 0.5])
    out = po._prox_quantile_identity(v, mu, y, tau=tau)
    np.testing.assert_allclose(out, [upper - 0.05, upper, lower], rtol=0, atol=1e-12)


def test_quantile_identity_with_weights() -> None:
    # Per-observation weights scale the pinball term; the prox should
    # pick that up by widening / narrowing the interior pass-through
    # region in the same closed-form way.
    from scipy.optimize import minimize_scalar

    mu = 1.0
    tau = 0.4
    v = np.array([0.2, -0.5, 1.0])
    y = np.array([2.0, -2.0, 0.5])
    w = np.array([0.5, 1.5, 1.0])
    out = po._prox_quantile_identity(v, mu, y, tau=tau, w=w)
    for i in range(len(v)):

        def obj(
            x: float, _v: float = v[i], _y: float = y[i], _w: float = w[i]
        ) -> float:
            return _w * _pinball(_y - x, tau) + 0.5 * mu * (x - _v) ** 2

        expected = float(minimize_scalar(obj).x)
        assert abs(out[i] - expected) < 1e-5


def test_quantile_median_is_symmetric_soft_threshold() -> None:
    # tau=0.5 collapses to the symmetric soft-threshold of the L1 prox,
    # with half-width 1/(2*mu) on each side of v.
    mu = 4.0
    v = np.array([0.0, 0.0, 0.0, 0.0])
    half = 0.5 / mu
    y = np.array([0.0, half - 1e-6, half + 1.0, -half - 1.0])
    out = po._prox_quantile_identity(v, mu, y, tau=0.5)
    np.testing.assert_allclose(out, [0.0, half - 1e-6, half, -half], atol=1e-12)
