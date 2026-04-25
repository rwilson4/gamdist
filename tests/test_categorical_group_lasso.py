"""Tests for the group_lasso regularization on _CategoricalFeature.

Group lasso on a categorical feature drives the entire per-level
parameter vector p toward zero, giving categorical-variable selection
(either every level is in the model or none is). The penalty term is
``lambda * ||A q||_2 = lambda * sqrt(q^T diag(ccs) q)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM
from gamdist.categorical_feature import _CategoricalFeature


def test_group_lasso_no_coef_raises() -> None:
    with pytest.raises(ValueError, match="No coefficient specified for group_lasso"):
        _CategoricalFeature(name="g", regularization={"group_lasso": {}})


def test_group_lasso_smoothing_scales_lambda() -> None:
    feat = _CategoricalFeature(
        name="g", regularization={"group_lasso": {"coef": 0.4}}
    )
    feat.initialize(np.array(["a", "b", "a"]), smoothing=2.5)
    assert feat._has_group_lasso
    assert feat._lambda_group_lasso == pytest.approx(1.0)


def test_group_lasso_lambda_zero_matches_unpenalized() -> None:
    rng = np.random.default_rng(0)
    x = rng.choice(np.array(["a", "b", "c"]), size=200)
    fpumz = rng.normal(size=200)

    plain = _CategoricalFeature(name="g")
    plain.initialize(x)
    plain.optimize(fpumz, rho=1.0)

    zero = _CategoricalFeature(name="g", regularization={"group_lasso": {"coef": 0.0}})
    zero.initialize(x)
    zero.optimize(fpumz, rho=1.0)

    np.testing.assert_allclose(zero.p, plain.p, atol=1e-6)


def test_group_lasso_huge_lambda_zeros_parameters() -> None:
    rng = np.random.default_rng(0)
    x = rng.choice(np.array(["a", "b", "c"]), size=200)
    fpumz = rng.normal(size=200)
    feat = _CategoricalFeature(
        name="g", regularization={"group_lasso": {"coef": 1e6}}
    )
    feat.initialize(x)
    feat.optimize(fpumz, rho=1.0)
    np.testing.assert_allclose(feat.p, 0.0, atol=1e-6)


def test_group_lasso_intermediate_lambda_shrinks() -> None:
    # Predictions should be strictly smaller in magnitude under a
    # moderate group lasso penalty than without.
    rng = np.random.default_rng(0)
    x = rng.choice(np.array(["a", "b", "c"]), size=200)
    fpumz = rng.normal(size=200) + 2.0  # strong signal so the unpenalized fit isn't tiny

    unpenalized = _CategoricalFeature(name="g")
    unpenalized.initialize(x)
    unpenalized.optimize(fpumz, rho=1.0)

    moderate = _CategoricalFeature(
        name="g", regularization={"group_lasso": {"coef": 0.5}}
    )
    moderate.initialize(x)
    moderate.optimize(fpumz, rho=1.0)

    assert np.linalg.norm(moderate.p) < np.linalg.norm(unpenalized.p)


def test_group_lasso_save_load_round_trip(tmp_path: Path) -> None:
    feat = _CategoricalFeature(
        name="g", regularization={"group_lasso": {"coef": 0.7}}
    )
    feat.initialize(
        np.array(["a", "b", "a"]),
        smoothing=2.0,
        save_flag=True,
        save_prefix=str(tmp_path / "model"),
    )
    feat.p = np.array([0.3, -0.3])
    feat._save()

    restored = _CategoricalFeature(load_from_file=feat._filename)
    assert restored._has_group_lasso
    assert restored._lambda_group_lasso == pytest.approx(1.4)
    np.testing.assert_allclose(restored.p, feat.p)


def test_group_lasso_within_gam_drops_noise_feature() -> None:
    # Two categorical features. "signal" actually drives y; "noise" does
    # not. With a moderate group lasso applied to *both*, noise should
    # shrink much more than signal.
    rng = np.random.default_rng(2)
    n = 400
    signal = rng.choice(np.array(["a", "b", "c"]), size=n)
    noise = rng.choice(np.array(["x", "y", "z"]), size=n)
    signal_effects = {"a": 1.5, "b": -1.0, "c": -0.5}
    y = np.array([signal_effects[s] for s in signal]) + rng.normal(size=n) * 0.1
    X = pd.DataFrame({"signal": signal, "noise": noise})

    mdl = GAM(family="normal")
    mdl.add_feature(
        name="signal",
        type="categorical",
        regularization={"group_lasso": {"coef": 0.3}},
    )
    mdl.add_feature(
        name="noise",
        type="categorical",
        regularization={"group_lasso": {"coef": 0.3}},
    )
    mdl.fit(X, y, max_its=40)

    signal_feat = mdl._features["signal"]
    noise_feat = mdl._features["noise"]
    signal_norm = float(np.linalg.norm(signal_feat.p))  # type: ignore[attr-defined]
    noise_norm = float(np.linalg.norm(noise_feat.p))  # type: ignore[attr-defined]
    # Signal feature should retain meaningful magnitude; noise should be
    # heavily shrunk relative to it.
    assert signal_norm > 0.5
    assert noise_norm < 0.1 * signal_norm
