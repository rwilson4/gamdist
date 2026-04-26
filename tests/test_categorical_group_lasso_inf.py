"""Tests for the ``group_lasso_inf`` regularization on _CategoricalFeature.

The L_inf-norm group lasso penalizes ``λ · ||A q||_inf = λ · max_c |q_c|``
(restricted to categories that actually appear in the data). It clips the
largest per-level effect rather than the uniform L2 contraction induced by
the standard group lasso, so the resulting shrinkage geometry is different.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM
from gamdist.categorical_feature import _CategoricalFeature


def test_group_lasso_inf_no_coef_raises() -> None:
    with pytest.raises(
        ValueError, match="No coefficient specified for group_lasso_inf"
    ):
        _CategoricalFeature(name="g", regularization={"group_lasso_inf": {}})


def test_group_lasso_inf_smoothing_scales_lambda() -> None:
    feat = _CategoricalFeature(
        name="g", regularization={"group_lasso_inf": {"coef": 0.4}}
    )
    feat.initialize(np.array(["a", "b", "a"]), smoothing=2.5)
    assert feat._has_group_lasso_inf
    assert feat._lambda_group_lasso_inf == pytest.approx(1.0)


def test_group_lasso_inf_lambda_zero_matches_unpenalized() -> None:
    rng = np.random.default_rng(0)
    x = rng.choice(np.array(["a", "b", "c"]), size=200)
    fpumz = rng.normal(size=200)

    plain = _CategoricalFeature(name="g")
    plain.initialize(x)
    plain.optimize(fpumz, rho=1.0)

    zero = _CategoricalFeature(
        name="g", regularization={"group_lasso_inf": {"coef": 0.0}}
    )
    zero.initialize(x)
    zero.optimize(fpumz, rho=1.0)

    np.testing.assert_allclose(zero.p, plain.p, atol=1e-6)


def test_group_lasso_inf_huge_lambda_zeros_parameters() -> None:
    rng = np.random.default_rng(0)
    x = rng.choice(np.array(["a", "b", "c"]), size=200)
    fpumz = rng.normal(size=200)
    feat = _CategoricalFeature(
        name="g", regularization={"group_lasso_inf": {"coef": 1e6}}
    )
    feat.initialize(x)
    feat.optimize(fpumz, rho=1.0)
    np.testing.assert_allclose(feat.p, 0.0, atol=1e-6)


def test_group_lasso_inf_clips_max_level() -> None:
    # Different shrinkage geometries: the L2 group lasso shrinks every
    # level roughly proportionally, while the L_inf variant clips only
    # the largest |q_c| and leaves the smaller levels close to their
    # unpenalized values. Construct fpumz so that one category has a
    # much larger unpenalized estimate than the others, then verify
    # both effects on the same data.
    rng = np.random.default_rng(11)
    n = 600
    cats = np.array(["a", "b", "c"])
    x = rng.choice(cats, size=n)
    fpumz = rng.normal(size=n) * 0.1
    fpumz[x == "a"] -= 3.0  # Inflate category "a"'s unpenalized effect.

    unpen = _CategoricalFeature(name="g")
    unpen.initialize(x)
    unpen.optimize(fpumz, rho=1.0)

    coef = 5.0
    linf = _CategoricalFeature(
        name="g", regularization={"group_lasso_inf": {"coef": coef}}
    )
    linf.initialize(x)
    linf.optimize(fpumz, rho=1.0)

    l2 = _CategoricalFeature(name="g", regularization={"group_lasso": {"coef": coef}})
    l2.initialize(x)
    l2.optimize(fpumz, rho=1.0)

    # L_inf reduces the maximum |q_c| relative to the unpenalized fit.
    assert np.max(np.abs(linf.p)) < np.max(np.abs(unpen.p))

    # Smaller-level effects are preserved more under L_inf than under L2.
    # Identify the non-extremal levels by ordering on |unpen.p|.
    order = np.argsort(np.abs(unpen.p))
    smaller = order[:-1]
    # Sum of |delta| from unpenalized over the non-max levels: L2 moves
    # them noticeably, L_inf barely touches them.
    l2_smaller_drift = float(np.sum(np.abs(l2.p[smaller] - unpen.p[smaller])))
    linf_smaller_drift = float(np.sum(np.abs(linf.p[smaller] - unpen.p[smaller])))
    assert linf_smaller_drift < l2_smaller_drift


def test_group_lasso_inf_combines_with_l2_variant() -> None:
    # Verify that having both variants on simultaneously runs to a valid
    # optimum and shrinks at least as much as either alone.
    rng = np.random.default_rng(4)
    x = rng.choice(np.array(["a", "b", "c", "d"]), size=300)
    fpumz = rng.normal(size=300)

    only_l2 = _CategoricalFeature(
        name="g", regularization={"group_lasso": {"coef": 0.5}}
    )
    only_l2.initialize(x)
    only_l2.optimize(fpumz, rho=1.0)

    only_linf = _CategoricalFeature(
        name="g", regularization={"group_lasso_inf": {"coef": 0.5}}
    )
    only_linf.initialize(x)
    only_linf.optimize(fpumz, rho=1.0)

    both = _CategoricalFeature(
        name="g",
        regularization={
            "group_lasso": {"coef": 0.5},
            "group_lasso_inf": {"coef": 0.5},
        },
    )
    both.initialize(x)
    both.optimize(fpumz, rho=1.0)

    # Stacking shrinks at least as hard as either alone (in both norms).
    assert np.linalg.norm(both.p) <= np.linalg.norm(only_l2.p) + 1e-6
    assert np.max(np.abs(both.p)) <= np.max(np.abs(only_linf.p)) + 1e-6


def test_group_lasso_inf_save_load_round_trip(tmp_path: Path) -> None:
    feat = _CategoricalFeature(
        name="g", regularization={"group_lasso_inf": {"coef": 0.7}}
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
    assert restored._has_group_lasso_inf
    assert restored._lambda_group_lasso_inf == pytest.approx(1.4)
    np.testing.assert_allclose(restored.p, feat.p)


def test_group_lasso_inf_within_gam_drops_noise_feature() -> None:
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
        regularization={"group_lasso_inf": {"coef": 0.3}},
    )
    mdl.add_feature(
        name="noise",
        type="categorical",
        regularization={"group_lasso_inf": {"coef": 0.3}},
    )
    mdl.fit(X, y, max_its=40)

    signal_feat = mdl._features["signal"]
    noise_feat = mdl._features["noise"]
    signal_norm = float(np.linalg.norm(signal_feat.p))  # type: ignore[attr-defined]
    noise_norm = float(np.linalg.norm(noise_feat.p))  # type: ignore[attr-defined]
    assert signal_norm > 0.5
    assert noise_norm < 0.2 * signal_norm
