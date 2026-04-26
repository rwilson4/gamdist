"""Unit tests for the _SplineFeature class and helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gamdist import SplineError
from gamdist.spline_feature import (
    _evaluate_spline_basis,
    _omega_curvature,
    _sample,
    _SplineFeature,
)


def test_sample_returns_endpoints() -> None:
    x = np.linspace(0.0, 1.0, 200)
    out = _sample(x)
    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(1.0)


def test_evaluate_spline_basis_requires_two_knots() -> None:
    with pytest.raises(SplineError, match="at least two knots"):
        _evaluate_spline_basis(0.5, np.array([0.5]))


def test_omega_curvature_requires_two_knots() -> None:
    with pytest.raises(SplineError, match="at least two knots"):
        _omega_curvature(np.array([0.5]))


def test_omega_is_symmetric() -> None:
    xi = np.linspace(0.0, 1.0, 10)
    omega = _omega_curvature(xi)
    np.testing.assert_allclose(omega, omega.T, atol=1e-12)


def test_initialize_and_predict_constant() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0.0, 1.0, size=n)
    feat = _SplineFeature(name="x", rel_dof=4.0)
    feat.initialize(x)
    pred = feat.predict(x)
    # Theta starts at zero, so prediction is zero before fitting.
    np.testing.assert_allclose(pred, 0.0)


def test_optimize_recovers_smooth_signal() -> None:
    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(0.0, 1.0, size=n)
    y = np.sin(2 * np.pi * x)
    y = y - y.mean()
    feat = _SplineFeature(name="x", rel_dof=8.0)
    feat.initialize(x)
    for _ in range(3):
        feat.optimize(-y, rho=1.0)
    pred = feat.predict(x)
    # Loose check: the fitted spline should be reasonably correlated with y.
    corr = np.corrcoef(pred, y)[0, 1]
    assert corr > 0.7


def test_save_load_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, size=100)
    feat = _SplineFeature(name="x", rel_dof=4.0)
    feat.initialize(x, save_flag=True, save_prefix=str(tmp_path / "model"))
    feat._theta = np.linspace(0.0, 1.0, len(feat._theta))
    feat._save()

    restored = _SplineFeature(load_from_file=feat._filename)
    np.testing.assert_allclose(restored._theta, feat._theta)
    np.testing.assert_allclose(restored._xi, feat._xi)
