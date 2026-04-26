"""Tests for the ``group_lasso_inf`` regularization on _SplineFeature.

For a spline feature the per-feature contribution vector is
``f_j = N θ``. The L_inf-norm group-lasso penalty ``λ · ||N θ||_inf``
clips the largest pointwise contribution rather than uniformly
contracting the whole curve. The closed-form Cholesky path is bypassed
in favor of the cvxpy path whenever this term is active.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM
from gamdist.spline_feature import _SplineFeature


def test_group_lasso_inf_no_coef_raises() -> None:
    with pytest.raises(
        ValueError, match="No coefficient specified for group_lasso_inf"
    ):
        _SplineFeature(name="x", regularization={"group_lasso_inf": {}})


def test_group_lasso_inf_smoothing_scales_lambda() -> None:
    feat = _SplineFeature(name="x", regularization={"group_lasso_inf": {"coef": 0.4}})
    feat.initialize(np.linspace(0.0, 1.0, 50), smoothing=2.5)
    assert feat._has_group_lasso_inf
    assert feat._lambda_group_lasso_inf == pytest.approx(1.0)


def test_group_lasso_inf_lambda_zero_matches_unpenalized() -> None:
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
        name="x", rel_dof=6.0, regularization={"group_lasso_inf": {"coef": 0.0}}
    )
    zero.initialize(x)
    zero.optimize(-y, rho=1.0)
    zero_pred = zero._N.dot(zero._theta)

    np.testing.assert_allclose(zero_pred, plain_pred, atol=1e-6)


def test_group_lasso_inf_huge_lambda_zeros_contribution() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0.0, 1.0, size=n)
    y = np.sin(2 * np.pi * x)
    y = y - y.mean()
    feat = _SplineFeature(
        name="x", rel_dof=6.0, regularization={"group_lasso_inf": {"coef": 1e3}}
    )
    feat.initialize(x)
    feat.optimize(-y, rho=1.0)
    pred = feat.predict(x)
    np.testing.assert_allclose(pred, 0.0, atol=1e-3)


def test_group_lasso_inf_intermediate_lambda_shrinks() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0.0, 1.0, size=n)
    y = 3.0 * np.sin(2 * np.pi * x)
    y = y - y.mean()

    unpenalized = _SplineFeature(name="x", rel_dof=6.0)
    unpenalized.initialize(x)
    unpenalized.optimize(-y, rho=1.0)

    moderate = _SplineFeature(
        name="x", rel_dof=6.0, regularization={"group_lasso_inf": {"coef": 1.0}}
    )
    moderate.initialize(x)
    moderate.optimize(-y, rho=1.0)

    # Both the L_inf norm and the L2 norm of the contribution shrink.
    assert np.max(np.abs(moderate.predict(x))) < np.max(np.abs(unpenalized.predict(x)))
    assert np.linalg.norm(moderate.predict(x)) < np.linalg.norm(unpenalized.predict(x))


def test_group_lasso_inf_combines_with_l2_variant() -> None:
    # Both variants on simultaneously: the optimization should still
    # converge and shrink the contribution at least as much (in both
    # norms) as either alone.
    rng = np.random.default_rng(3)
    n = 200
    x = rng.uniform(0.0, 1.0, size=n)
    y = 2.0 * np.sin(2 * np.pi * x)
    y = y - y.mean()

    only_l2 = _SplineFeature(
        name="x", rel_dof=6.0, regularization={"group_lasso": {"coef": 0.5}}
    )
    only_l2.initialize(x)
    only_l2.optimize(-y, rho=1.0)

    only_linf = _SplineFeature(
        name="x", rel_dof=6.0, regularization={"group_lasso_inf": {"coef": 0.5}}
    )
    only_linf.initialize(x)
    only_linf.optimize(-y, rho=1.0)

    both = _SplineFeature(
        name="x",
        rel_dof=6.0,
        regularization={
            "group_lasso": {"coef": 0.5},
            "group_lasso_inf": {"coef": 0.5},
        },
    )
    both.initialize(x)
    both.optimize(-y, rho=1.0)

    assert np.linalg.norm(both.predict(x)) <= np.linalg.norm(only_l2.predict(x)) + 1e-3
    assert (
        np.max(np.abs(both.predict(x))) <= np.max(np.abs(only_linf.predict(x))) + 1e-3
    )


def test_group_lasso_inf_save_load_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, size=50)
    feat = _SplineFeature(
        name="x", rel_dof=4.0, regularization={"group_lasso_inf": {"coef": 0.7}}
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
    assert restored._has_group_lasso_inf
    assert restored._coef_group_lasso_inf == pytest.approx(0.7)
    assert restored._lambda_group_lasso_inf == pytest.approx(1.4)
    np.testing.assert_allclose(restored._theta, feat._theta)


def test_group_lasso_inf_within_gam_drops_noise_feature() -> None:
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
        regularization={"group_lasso_inf": {"coef": 0.3}},
    )
    mdl.add_feature(
        name="noise",
        type="spline",
        rel_dof=6.0,
        regularization={"group_lasso_inf": {"coef": 0.3}},
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
