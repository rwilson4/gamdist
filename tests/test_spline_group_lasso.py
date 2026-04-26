"""Tests for the group_lasso regularization on _SplineFeature.

For a spline feature the per-feature contribution vector is
``f_j = N θ``. The group-lasso penalty ``λ · ||N θ||_2`` shrinks the
entire fitted spline toward zero and can zero it out completely once
λ is large enough. The closed-form Cholesky path is preserved when
group lasso is off; this test file targets the cvxpy-based path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM
from gamdist.spline_feature import _SplineFeature


def test_group_lasso_no_coef_raises() -> None:
    with pytest.raises(ValueError, match="No coefficient specified for group_lasso"):
        _SplineFeature(name="x", regularization={"group_lasso": {}})


def test_group_lasso_smoothing_scales_lambda() -> None:
    feat = _SplineFeature(name="x", regularization={"group_lasso": {"coef": 0.4}})
    feat.initialize(np.linspace(0.0, 1.0, 50), smoothing=2.5)
    assert feat._has_group_lasso
    assert feat._lambda_group_lasso == pytest.approx(1.0)


def test_group_lasso_lambda_zero_matches_unpenalized() -> None:
    # NtN is heavily rank-deficient (the cubic-regression-spline basis on
    # finitely many distinct x values has nontrivial nullspace), so the
    # raw theta vector isn't unique. The fitted contribution N @ theta is.
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0.0, 1.0, size=n)
    y = np.sin(2 * np.pi * x)
    y = y - y.mean()

    plain = _SplineFeature(name="x", rel_dof=6.0)
    plain.initialize(x)
    plain.optimize(-y, rho=1.0)
    plain_pred = plain._N.dot(plain._theta)

    zero = _SplineFeature(
        name="x", rel_dof=6.0, regularization={"group_lasso": {"coef": 0.0}}
    )
    zero.initialize(x)
    zero.optimize(-y, rho=1.0)
    zero_pred = zero._N.dot(zero._theta)

    np.testing.assert_allclose(zero_pred, plain_pred, atol=1e-6)


def test_group_lasso_huge_lambda_zeros_contribution() -> None:
    # CLARABEL becomes numerically unstable at very large penalty
    # multipliers (~1e6 on the SOC coefficient is too aggressive for the
    # default tolerances), so use a still-large-but-tractable λ that's
    # well above the soft-threshold boundary for the data scale.
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0.0, 1.0, size=n)
    y = np.sin(2 * np.pi * x)
    y = y - y.mean()
    feat = _SplineFeature(
        name="x", rel_dof=6.0, regularization={"group_lasso": {"coef": 1e3}}
    )
    feat.initialize(x)
    feat.optimize(-y, rho=1.0)
    pred = feat.predict(x)
    np.testing.assert_allclose(pred, 0.0, atol=1e-4)


def test_group_lasso_intermediate_lambda_shrinks() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0.0, 1.0, size=n)
    y = 3.0 * np.sin(2 * np.pi * x)
    y = y - y.mean()

    unpenalized = _SplineFeature(name="x", rel_dof=6.0)
    unpenalized.initialize(x)
    unpenalized.optimize(-y, rho=1.0)

    moderate = _SplineFeature(
        name="x", rel_dof=6.0, regularization={"group_lasso": {"coef": 1.0}}
    )
    moderate.initialize(x)
    moderate.optimize(-y, rho=1.0)

    assert np.linalg.norm(moderate.predict(x)) < np.linalg.norm(unpenalized.predict(x))


def test_group_lasso_save_load_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, size=50)
    feat = _SplineFeature(
        name="x", rel_dof=4.0, regularization={"group_lasso": {"coef": 0.7}}
    )
    feat.initialize(
        x,
        smoothing=2.0,
        save_flag=True,
        save_prefix=str(tmp_path / "model"),
    )
    feat._theta = np.linspace(0.0, 1.0, len(feat._theta))
    feat._save()

    restored = _SplineFeature(load_from_file=feat._filename)
    assert restored._has_group_lasso
    assert restored._coef_group_lasso == pytest.approx(0.7)
    assert restored._lambda_group_lasso == pytest.approx(1.4)
    np.testing.assert_allclose(restored._theta, feat._theta)


def test_group_lasso_within_gam_drops_noise_feature() -> None:
    # Two spline features. "signal" actually drives y; "noise" does not.
    # With a moderate group lasso applied to *both*, the noise spline
    # should shrink heavily while the signal spline retains a real shape.
    rng = np.random.default_rng(2)
    n = 400
    signal = rng.uniform(0.0, 1.0, size=n)
    noise = rng.uniform(0.0, 1.0, size=n)
    y = np.sin(2 * np.pi * signal) + 0.1 * rng.normal(size=n)
    y = y - y.mean()
    X = pd.DataFrame({"signal": signal, "noise": noise})

    mdl = GAM(family="normal")
    mdl.add_feature(
        name="signal",
        type="spline",
        rel_dof=6.0,
        regularization={"group_lasso": {"coef": 0.3}},
    )
    mdl.add_feature(
        name="noise",
        type="spline",
        rel_dof=6.0,
        regularization={"group_lasso": {"coef": 0.3}},
    )
    mdl.fit(X, y, max_its=40)

    signal_feat = mdl._features["signal"]
    noise_feat = mdl._features["noise"]
    signal_pred = signal_feat.predict(signal)
    noise_pred = noise_feat.predict(noise)
    signal_norm = float(np.linalg.norm(signal_pred))
    noise_norm = float(np.linalg.norm(noise_pred))
    assert signal_norm > 1.0
    assert noise_norm < 0.2 * signal_norm
