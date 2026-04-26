"""Tests for the two-stage adaptive lasso wrapper.

Adaptive lasso (Zou 2006) refits a model after rewriting each L1
coefficient to ``base / (|pilot| + eps) ** gamma``. Coefficients with
large pilot magnitudes are penalized less; tiny pilots are penalized
more. The expected behavior on a sparse-truth signal is:

  * Noise features shrink harder than they would under plain L1.
  * Signal features are biased less toward zero than they would be under
    plain L1, recovering the pilot-driven "oracle" estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM, fit_adaptive_lasso
from gamdist.categorical_feature import _CategoricalFeature
from gamdist.linear_feature import _LinearFeature


def _build_linear_gam(coef: float) -> GAM:
    mdl = GAM(family="normal")
    mdl.add_feature(name="signal", type="linear", regularization={"l1": {"coef": coef}})
    mdl.add_feature(name="noise1", type="linear", regularization={"l1": {"coef": coef}})
    mdl.add_feature(name="noise2", type="linear", regularization={"l1": {"coef": coef}})
    return mdl


def _make_sparse_linear_data(
    seed: int = 0, n: int = 400
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    noise1 = rng.normal(size=n)
    noise2 = rng.normal(size=n)
    y = 2.0 * (signal - signal.mean()) + 0.1 * rng.normal(size=n)
    X = pd.DataFrame({"signal": signal, "noise1": noise1, "noise2": noise2})
    return X, y


def test_validates_gamma_and_eps() -> None:
    X, y = _make_sparse_linear_data()
    mdl = _build_linear_gam(coef=0.3)
    with pytest.raises(ValueError, match="gamma must be positive"):
        fit_adaptive_lasso(mdl, X, y, gamma=0.0, max_its=5)
    with pytest.raises(ValueError, match="gamma must be positive"):
        fit_adaptive_lasso(mdl, X, y, gamma=-1.0, max_its=5)
    with pytest.raises(ValueError, match="eps must be positive"):
        fit_adaptive_lasso(mdl, X, y, eps=0.0, max_its=5)


def test_requires_at_least_one_l1_feature() -> None:
    X, y = _make_sparse_linear_data()
    mdl = GAM(family="normal")
    mdl.add_feature(name="signal", type="linear")
    mdl.add_feature(name="noise1", type="linear", regularization={"l2": {"coef": 0.5}})
    with pytest.raises(ValueError, match="requires at least one feature with"):
        fit_adaptive_lasso(mdl, pd.concat([X[["signal", "noise1"]]], axis=1), y)


def test_linear_apply_adaptive_l1_rewrites_coef1() -> None:
    feat = _LinearFeature(name="x", regularization={"l1": {"coef": 0.3}})
    feat.initialize(np.arange(10.0))
    feat._m = 0.5
    rewrote = feat._apply_adaptive_l1(gamma=1.0, eps=1e-6)
    assert rewrote
    assert feat._coef1 == pytest.approx(0.3 / (0.5 + 1e-6))


def test_linear_apply_adaptive_l1_skips_when_no_l1() -> None:
    feat = _LinearFeature(name="x", regularization={"l2": {"coef": 0.5}})
    feat.initialize(np.arange(10.0))
    feat._m = 0.5
    assert feat._apply_adaptive_l1(gamma=1.0, eps=1e-6) is False


def test_linear_apply_adaptive_l1_skips_zero_base_coef() -> None:
    feat = _LinearFeature(name="x", regularization={"l1": {"coef": 0.0}})
    feat.initialize(np.arange(10.0))
    feat._m = 0.5
    assert feat._apply_adaptive_l1(gamma=1.0, eps=1e-6) is False


def test_categorical_apply_adaptive_l1_uses_dict_form() -> None:
    feat = _CategoricalFeature(name="g", regularization={"l1": {"coef": 0.5}})
    feat.initialize(np.array(["a", "b", "c"]))
    a, b, c = (feat._category_hash[k] for k in ("a", "b", "c"))
    feat.p = np.zeros(3)
    feat.p[a] = 1.5
    feat.p[b] = 0.1
    feat.p[c] = -0.05
    rewrote = feat._apply_adaptive_l1(gamma=1.0, eps=1e-6)
    assert rewrote
    assert isinstance(feat._coef1, dict)
    # Larger pilot magnitude => smaller penalty
    assert feat._coef1["a"] < feat._coef1["b"]
    assert feat._coef1["a"] < feat._coef1["c"]
    assert feat._coef1["a"] == pytest.approx(0.5 / (1.5 + 1e-6))
    assert feat._coef1["b"] == pytest.approx(0.5 / (0.1 + 1e-6))


def test_categorical_apply_adaptive_l1_with_dict_base_only_touches_listed_cats() -> (
    None
):
    feat = _CategoricalFeature(
        name="g", regularization={"l1": {"coef": {"a": 0.4, "b": 0.6}}}
    )
    feat.initialize(np.array(["a", "b", "c"]))
    a, b, c = (feat._category_hash[k] for k in ("a", "b", "c"))
    feat.p = np.zeros(3)
    feat.p[a] = 0.5
    feat.p[b] = 1.0
    feat.p[c] = 2.0
    feat._apply_adaptive_l1(gamma=1.0, eps=1e-6)
    assert isinstance(feat._coef1, dict)
    assert "c" not in feat._coef1
    assert feat._coef1["a"] == pytest.approx(0.4 / (0.5 + 1e-6))
    assert feat._coef1["b"] == pytest.approx(0.6 / (1.0 + 1e-6))


def test_categorical_apply_adaptive_l1_with_prior_uses_deviation() -> None:
    feat = _CategoricalFeature(
        name="g",
        regularization={"l1": {"coef": 0.5}, "prior": {"a": 1.0, "b": -1.0}},
    )
    feat.initialize(np.array(["a", "b", "c"]))
    a, b, c = (feat._category_hash[k] for k in ("a", "b", "c"))
    feat.p = np.zeros(3)
    feat.p[a] = 1.2  # deviation 0.2 from prior 1.0
    feat.p[b] = -0.5  # deviation 0.5 from prior -1.0
    feat.p[c] = 5.0  # not in prior keys; should be skipped
    feat._apply_adaptive_l1(gamma=1.0, eps=1e-6)
    assert isinstance(feat._coef1, dict)
    # c was not in prior keys -> excluded from new coef1
    assert "c" not in feat._coef1
    assert feat._coef1["a"] == pytest.approx(0.5 / (0.2 + 1e-6))
    assert feat._coef1["b"] == pytest.approx(0.5 / (0.5 + 1e-6))


def test_linear_adaptive_lasso_shrinks_noise_more_than_plain() -> None:
    """End-to-end: adaptive lasso shrinks noise and protects signal more
    than plain lasso at the same base coefficient."""
    X, y = _make_sparse_linear_data(seed=2, n=400)

    plain = _build_linear_gam(coef=0.3)
    plain.fit(X, y, max_its=80)
    plain_signal = abs(plain._features["signal"]._m)  # type: ignore[attr-defined]
    plain_noise1 = abs(plain._features["noise1"]._m)  # type: ignore[attr-defined]
    plain_noise2 = abs(plain._features["noise2"]._m)  # type: ignore[attr-defined]

    adapted = _build_linear_gam(coef=0.3)
    fit_adaptive_lasso(adapted, X, y, gamma=1.0, eps=1e-3, max_its=80)
    adapt_signal = abs(adapted._features["signal"]._m)  # type: ignore[attr-defined]
    adapt_noise1 = abs(adapted._features["noise1"]._m)  # type: ignore[attr-defined]
    adapt_noise2 = abs(adapted._features["noise2"]._m)  # type: ignore[attr-defined]

    # Noise shrinks at least as much under adaptive
    assert adapt_noise1 <= plain_noise1 + 1e-9
    assert adapt_noise2 <= plain_noise2 + 1e-9
    # Signal is biased less under adaptive (i.e., closer to true 2.0)
    true_signal = 2.0
    assert abs(adapt_signal - true_signal) < abs(plain_signal - true_signal)


def test_linear_adaptive_lasso_drives_pure_noise_features_to_zero() -> None:
    """With gamma=1 and a small eps, noise features whose pilot is already
    near zero should round-trip to essentially zero in stage 2."""
    X, y = _make_sparse_linear_data(seed=5, n=300)
    mdl = _build_linear_gam(coef=0.4)
    fit_adaptive_lasso(mdl, X, y, gamma=1.0, eps=1e-3, max_its=80)
    # Signal should remain substantial; both noise features should vanish.
    signal_m = abs(mdl._features["signal"]._m)  # type: ignore[attr-defined]
    noise1_m = abs(mdl._features["noise1"]._m)  # type: ignore[attr-defined]
    noise2_m = abs(mdl._features["noise2"]._m)  # type: ignore[attr-defined]
    assert signal_m > 1.0
    assert noise1_m == pytest.approx(0.0, abs=1e-6)
    assert noise2_m == pytest.approx(0.0, abs=1e-6)


def test_categorical_adaptive_lasso_recovers_sparse_levels() -> None:
    """Categorical with 5 levels where only 2 carry non-zero true effects.
    Adaptive lasso should shrink the three null levels harder than plain
    lasso, while keeping the active levels close to truth."""
    rng = np.random.default_rng(7)
    n = 800
    cats = rng.choice(["a", "b", "c", "d", "e"], size=n)
    true_effects = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 1.0, "e": -1.0}
    eta = np.array([true_effects[c] for c in cats]) + 0.2 * rng.normal(size=n)
    X = pd.DataFrame({"g": cats})

    plain = GAM(family="normal")
    plain.add_feature(
        name="g", type="categorical", regularization={"l1": {"coef": 0.5}}
    )
    plain.fit(X, eta, max_its=80)
    plain_p = plain._features["g"].p.copy()  # type: ignore[attr-defined]
    plain_hash = plain._features["g"]._category_hash  # type: ignore[attr-defined]

    adapted = GAM(family="normal")
    adapted.add_feature(
        name="g", type="categorical", regularization={"l1": {"coef": 0.5}}
    )
    fit_adaptive_lasso(adapted, X, eta, gamma=1.0, eps=1e-3, max_its=80)
    adapt_p = adapted._features["g"].p  # type: ignore[attr-defined]
    adapt_hash = adapted._features["g"]._category_hash  # type: ignore[attr-defined]

    null_cats = ["a", "b", "c"]
    plain_null_norm = sum(abs(plain_p[plain_hash[c]]) for c in null_cats)
    adapt_null_norm = sum(abs(adapt_p[adapt_hash[c]]) for c in null_cats)
    # Adaptive lasso's stage-2 weights are huge for tiny pilots, so the
    # null categories shrink harder.
    assert adapt_null_norm < plain_null_norm

    # Active categories still close to truth (and zero-sum is enforced
    # globally by the categorical optimizer).
    assert adapt_p[adapt_hash["d"]] > 0.7
    assert adapt_p[adapt_hash["e"]] < -0.7
