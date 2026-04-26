"""Tests for non-canonical-link prox operators (the scipy.minimize_scalar paths)."""

from __future__ import annotations

import numpy as np
import pytest

from gamdist import proximal_operators as po


def _identity(x: float) -> float:
    return x


def _logit_inv(x: float) -> float:
    return float(np.exp(x) / (1.0 + np.exp(x)))


def _exp(x: float) -> float:
    return float(np.exp(x))


def _shifted_reciprocal(x: float) -> float:
    return 1.0 / (1.0 + abs(x))


def test_prox_normal_non_canonical_link() -> None:
    v = np.array([1.0, 2.0])
    y = np.array([0.5, 1.5])
    out = po._prox_normal(v, mu=1.0, y=y, inv_link=_identity)
    assert out.shape == v.shape
    assert np.all(np.isfinite(out))


def test_prox_normal_non_canonical_with_weights() -> None:
    v = np.array([1.0, 2.0])
    y = np.array([0.5, 1.5])
    w = np.array([1.0, 2.0])
    out = po._prox_normal(v, mu=1.0, y=y, w=w, inv_link=_identity)
    assert out.shape == v.shape


def test_prox_binomial_non_canonical_probit_like() -> None:
    v = np.array([0.0, 0.5])
    y = np.array([0.0, 1.0])
    out = po._prox_binomial(v, mu=2.0, y=y, inv_link=_logit_inv)
    assert out.shape == v.shape
    assert np.all(np.isfinite(out))


def test_prox_binomial_non_canonical_with_weights() -> None:
    v = np.array([0.0, 0.5])
    y = np.array([0.0, 1.0])
    w = np.array([1.0, 2.0])
    out = po._prox_binomial(v, mu=2.0, y=y, w=w, inv_link=_logit_inv)
    assert out.shape == v.shape


def test_prox_poisson_non_canonical() -> None:
    v = np.array([0.5, 1.0])
    y = np.array([1.0, 2.0])
    out = po._prox_poisson(v, mu=2.0, y=y, inv_link=_exp)
    assert out.shape == v.shape


def test_prox_poisson_non_canonical_with_weights() -> None:
    v = np.array([0.5, 1.0])
    y = np.array([1.0, 2.0])
    w = np.array([1.0, 2.0])
    out = po._prox_poisson(v, mu=2.0, y=y, w=w, inv_link=_exp)
    assert out.shape == v.shape


def test_prox_poisson_log_with_weights() -> None:
    v = np.array([0.5, 1.0])
    y = np.array([1.0, 2.0])
    w = np.array([1.0, 2.0])
    out = po._prox_poisson_log(v, mu=2.0, y=y, w=w)
    assert out.shape == v.shape


def test_prox_gamma_non_canonical() -> None:
    v = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    out = po._prox_gamma(v, mu=1.0, y=y, inv_link=_exp)
    assert out.shape == v.shape


def test_prox_gamma_non_canonical_with_weights() -> None:
    v = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    w = np.array([1.0, 2.0])
    out = po._prox_gamma(v, mu=1.0, y=y, w=w, inv_link=_exp)
    assert out.shape == v.shape


def test_prox_gamma_reciprocal_with_weights() -> None:
    v = np.array([0.5, 1.0])
    y = np.array([1.0, 2.0])
    w = np.array([1.0, 2.0])
    out = po._prox_gamma_reciprocal(v, mu=1.0, y=y, w=w)
    assert out.shape == v.shape


def test_prox_inv_gaussian_non_canonical() -> None:
    v = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    out = po._prox_inv_gaussian(v, mu=1.0, y=y, inv_link=_shifted_reciprocal)
    assert out.shape == v.shape


def test_prox_inv_gaussian_non_canonical_with_weights() -> None:
    v = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    w = np.array([1.0, 2.0])
    out = po._prox_inv_gaussian(v, mu=1.0, y=y, w=w, inv_link=_shifted_reciprocal)
    assert out.shape == v.shape


def test_prox_inv_gaussian_reciprocal_squared_with_weights() -> None:
    v = np.array([0.5, 1.0])
    y = np.array([1.0, 2.0])
    w = np.array([1.0, 2.0])
    out = po._prox_inv_gaussian_reciprocal_squared(v, mu=1.0, y=y, w=w)
    assert out.shape == v.shape


def test_prox_inv_gaussian_reciprocal_squared_robust_across_grid() -> None:
    """The eta-space Newton solver should never raise on a representative
    input grid; the previous undamped z-space Newton could crash with
    'Dual variable update failed to converge.' on harder inputs."""
    grid_v = [0.01, 0.1, 1.0, 10.0, 100.0]
    grid_mu = [0.1, 1.0, 10.0]
    grid_y = [0.01, 0.1, 1.0, 10.0]
    for v in grid_v:
        for mu in grid_mu:
            for y in grid_y:
                z2 = po._prox_inv_gaussian_reciprocal_squared_scalar([v, mu, y])
                assert np.isfinite(z2)
                assert z2 > 0.0


def test_prox_inv_gaussian_reciprocal_squared_matches_eta_minimization() -> None:
    """The prox is convex in eta = z^2; verify the z-space solver returns the
    same eta as a direct convex minimization in eta."""
    from scipy.optimize import minimize_scalar

    cases = [
        (0.5, 1.0, 1.0, None),
        (1.0, 2.0, 0.5, None),
        (2.0, 1.0, 0.1, 1.5),
        (0.1, 5.0, 2.0, None),
    ]
    for v, mu, y, w in cases:
        w_eff = 1.0 if w is None else w

        # Convex-in-eta objective.
        def obj_eta(
            eta: float, _v: float = v, _mu: float = mu, _y: float = y, _w: float = w_eff
        ) -> float:
            if eta <= 0.0:
                return float("inf")
            return (
                0.5 * _w * _y * eta
                - _w * np.sqrt(eta)
                + 0.5 * _mu * (eta - _v) * (eta - _v)
            )

        eta_ref = float(
            minimize_scalar(obj_eta, bounds=(1e-12, 1e6), method="bounded").x
        )
        xx = [v, mu, y] if w is None else [v, mu, y, w]
        eta_solver = po._prox_inv_gaussian_reciprocal_squared_scalar(xx)
        np.testing.assert_allclose(eta_solver, eta_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "fn",
    [
        po._prox_normal,
        po._prox_binomial,
        po._prox_poisson,
        po._prox_gamma,
        po._prox_inv_gaussian,
    ],
)
def test_non_canonical_prox_raises_when_inv_link_missing(fn) -> None:  # type: ignore[no-untyped-def]
    v = np.array([0.5, 1.0])
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="inv_link"):
        fn(v, mu=1.0, y=y)
