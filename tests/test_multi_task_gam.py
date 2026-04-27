"""Tests for ``MultiTaskGAM`` and ``_MultiTaskLinearFeature``.

The first multi-task slice (issue #39) covers:

- Per-task ``(family, link)`` heterogeneity (normal+identity and
  binomial+logit verified together).
- One coupling penalty: group-lasso across tasks on a linear feature,
  which zeros the feature simultaneously across all tasks once λ is
  large enough.
- Different ``n_k`` per task (the multi-task ADMM holds per-task
  primal/dual state, so unequal lengths fall out for free).

Persistence, summary/plot/CI/AIC paths are intentionally
``NotImplementedError`` in this slice; those are exercised below to
guard against accidental wiring-up without tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.special as special

from gamdist import GAM, MultiTaskGAM
from gamdist.multi_task_features import _MultiTaskLinearFeature

# --- _MultiTaskLinearFeature unit tests ---


def test_feature_rejects_zero_tasks() -> None:
    with pytest.raises(ValueError, match="num_tasks must be >= 1"):
        _MultiTaskLinearFeature("x", num_tasks=0)


def test_feature_rejects_unknown_regularization() -> None:
    with pytest.raises(ValueError, match="unsupported regularization"):
        _MultiTaskLinearFeature("x", num_tasks=2, regularization={"l1": {"coef": 0.1}})


def test_feature_requires_coef_for_group_lasso() -> None:
    with pytest.raises(ValueError, match="No coefficient specified"):
        _MultiTaskLinearFeature(
            "x", num_tasks=2, regularization={"group_lasso_across_tasks": {}}
        )


def test_feature_rejects_negative_coef() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        _MultiTaskLinearFeature(
            "x",
            num_tasks=2,
            regularization={"group_lasso_across_tasks": {"coef": -0.1}},
        )


def test_feature_smoothing_scales_lambda() -> None:
    feat = _MultiTaskLinearFeature(
        "x",
        num_tasks=2,
        regularization={"group_lasso_across_tasks": {"coef": 0.4}},
    )
    feat.initialize_multi(
        [np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0, 3.0])],
        smoothing=2.5,
    )
    assert feat._has_group_lasso_across_tasks
    assert feat._lambda_group_lasso_across_tasks == pytest.approx(1.0)


def test_feature_huge_lambda_zeros_all_slopes() -> None:
    rng = np.random.default_rng(0)
    n = 100
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    fpumz1 = rng.normal(size=n)
    fpumz2 = rng.normal(size=n)

    feat = _MultiTaskLinearFeature(
        "x",
        num_tasks=2,
        regularization={"group_lasso_across_tasks": {"coef": 1e6}},
    )
    feat.initialize_multi([x1, x2])
    feat.optimize_multi([fpumz1, fpumz2], rho=1.0)
    assert np.all(feat._m == 0.0)


def test_feature_lambda_zero_decouples_tasks() -> None:
    """With λ=0, each task's slope should match running the K subproblems
    independently as ordinary least-squares.
    """
    rng = np.random.default_rng(1)
    n1, n2 = 80, 120
    x1 = rng.normal(size=n1)
    x2 = rng.normal(size=n2)
    fpumz1 = rng.normal(size=n1)
    fpumz2 = rng.normal(size=n2)

    feat = _MultiTaskLinearFeature(
        "x",
        num_tasks=2,
        regularization={"group_lasso_across_tasks": {"coef": 0.0}},
    )
    feat.initialize_multi([x1, x2])
    feat.optimize_multi([fpumz1, fpumz2], rho=1.0)

    # Closed form for each task in isolation: m_k = -(x_k . fpumz_k) / (x_k . x_k)
    # because the warm-start uses self._m=0 so y = -fpumz.
    for k, (x, fpumz) in enumerate([(x1, fpumz1), (x2, fpumz2)]):
        x_c = x - x.mean()
        b = float(x_c.dot(-fpumz))
        expected = b / float(x_c.dot(x_c))
        assert feat._m[k] == pytest.approx(expected, rel=1e-10, abs=1e-10)


# --- MultiTaskGAM construction tests ---


def test_construction_rejects_empty_families() -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        MultiTaskGAM(families=[])  # type: ignore[arg-type]


def test_construction_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="not supported"):
        MultiTaskGAM(families=["bogus"])  # type: ignore[list-item]


def test_construction_rejects_quantile_family() -> None:
    with pytest.raises(NotImplementedError, match="quantile"):
        MultiTaskGAM(families=["normal", "quantile"])


def test_construction_rejects_mismatched_links_length() -> None:
    with pytest.raises(ValueError, match="length"):
        MultiTaskGAM(families=["normal", "normal"], links=["identity"])


def test_construction_rejects_unsupported_pair() -> None:
    with pytest.raises(ValueError, match="not a supported"):
        # binomial + identity is not in SUPPORTED_FAMILY_LINK_PAIRS
        MultiTaskGAM(families=["binomial"], links=["identity"])


def test_construction_defaults_to_canonical_links() -> None:
    mt = MultiTaskGAM(families=["normal", "binomial", "poisson"])
    assert mt._links == ["identity", "logistic", "log"]
    assert mt._known_dispersion == [False, True, True]


def test_construction_per_task_dispersions() -> None:
    mt = MultiTaskGAM(families=["normal", "normal"], dispersions=[1.0, None])
    assert mt._known_dispersion == [True, False]
    assert mt._dispersion[0] == 1.0


# --- MultiTaskGAM end-to-end fit tests ---


def _gen_two_task_normal(
    n1: int = 250, n2: int = 350, seed: int = 0
) -> tuple[list[pd.DataFrame], list[np.ndarray], dict[str, tuple[float, float]]]:
    """Generate two normal+identity tasks with different lengths.

    Returns the design matrices, responses, and the true per-feature
    per-task slopes so tests can assert recovery.
    """
    rng = np.random.default_rng(seed)
    truth = {"a": (1.5, 0.8), "b": (-0.7, 0.3)}
    a1 = rng.normal(size=n1)
    b1 = rng.normal(size=n1)
    a2 = rng.normal(size=n2)
    b2 = rng.normal(size=n2)
    y1 = truth["a"][0] * a1 + truth["b"][0] * b1 + 0.05 * rng.normal(size=n1)
    y2 = truth["a"][1] * a2 + truth["b"][1] * b2 + 0.05 * rng.normal(size=n2)
    return (
        [pd.DataFrame({"a": a1, "b": b1}), pd.DataFrame({"a": a2, "b": b2})],
        [y1, y2],
        truth,
    )


def test_fit_unequal_task_lengths_recovers_truth() -> None:
    Xs, ys, truth = _gen_two_task_normal()
    mt = MultiTaskGAM(families=["normal", "normal"])
    mt.add_feature("a", type="linear")
    mt.add_feature("b", type="linear")
    mt.fit(Xs, ys, max_its=80)

    assert mt._fitted
    # n_k differs: convergence trace should still be a single sequence
    # (max-across-tasks rule), with the final iterate below tolerance.
    assert mt.prim_res[-1] < mt.prim_tol[-1]
    assert mt.dual_res[-1] < mt.dual_tol[-1]

    a = mt._features["a"]
    b = mt._features["b"]
    assert a._m[0] == pytest.approx(truth["a"][0], abs=0.05)
    assert a._m[1] == pytest.approx(truth["a"][1], abs=0.05)
    assert b._m[0] == pytest.approx(truth["b"][0], abs=0.05)
    assert b._m[1] == pytest.approx(truth["b"][1], abs=0.05)


def test_fit_no_coupling_matches_independent_gams() -> None:
    """λ=0 (or no coupling) should give per-task slopes within ADMM
    tolerance of two separately-fit single-task GAMs.
    """
    Xs, ys, _ = _gen_two_task_normal(seed=4)

    # Independent baseline
    independents = []
    for X, y in zip(Xs, ys, strict=True):
        m = GAM(family="normal")
        m.add_feature("a", type="linear")
        m.add_feature("b", type="linear")
        m.fit(X, y, max_its=200)
        independents.append(m)

    mt = MultiTaskGAM(families=["normal", "normal"])
    mt.add_feature("a", type="linear")
    mt.add_feature("b", type="linear")
    mt.fit(Xs, ys, max_its=200)

    for k, ind in enumerate(independents):
        assert mt._features["a"]._m[k] == pytest.approx(
            ind._features["a"]._m,
            abs=2e-2,  # type: ignore[attr-defined]
        )
        assert mt._features["b"]._m[k] == pytest.approx(
            ind._features["b"]._m,
            abs=2e-2,  # type: ignore[attr-defined]
        )


def test_group_lasso_drops_noise_across_all_tasks() -> None:
    """The headline multi-task property: group-lasso across tasks zeros a
    noise feature uniformly in *both* tasks, even though running each
    task independently might leave a small spurious slope.
    """
    rng = np.random.default_rng(2)
    n = 400
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    y1 = 1.5 * signal + 0.1 * rng.normal(size=n)
    y2 = -0.8 * signal + 0.1 * rng.normal(size=n)
    X = pd.DataFrame({"signal": signal, "noise": noise})

    mt = MultiTaskGAM(families=["normal", "normal"])
    mt.add_feature(
        "signal",
        type="linear",
        regularization={"group_lasso_across_tasks": {"coef": 0.3}},
    )
    mt.add_feature(
        "noise",
        type="linear",
        regularization={"group_lasso_across_tasks": {"coef": 0.3}},
    )
    mt.fit([X, X], [y1, y2], max_its=80)

    sig = mt._features["signal"]
    noi = mt._features["noise"]
    assert abs(sig._m[0]) > 0.5
    assert abs(sig._m[1]) > 0.3
    # Both noise slopes must be exactly zero -- group-lasso prox zeros
    # the entire K-vector, not each entry independently.
    assert noi._m[0] == 0.0
    assert noi._m[1] == 0.0


def test_fit_mixed_normal_and_binomial() -> None:
    """Per-task (family, link): one task normal+identity, one task
    binomial+logit. Both should converge and predict in plausible ranges.
    """
    rng = np.random.default_rng(7)
    n = 500
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y_norm = 1.5 * a - 0.7 * b + 0.1 * rng.normal(size=n)
    eta = 0.8 * a + 0.3 * b
    y_bin = rng.binomial(1, special.expit(eta)).astype(float)
    X = pd.DataFrame({"a": a, "b": b})

    mt = MultiTaskGAM(families=["normal", "binomial"])
    mt.add_feature("a", type="linear")
    mt.add_feature("b", type="linear")
    mt.fit([X, X], [y_norm, y_bin], max_its=120)

    assert mt._fitted
    # Task 0 (normal) recovers true slopes well.
    assert mt._features["a"]._m[0] == pytest.approx(1.5, abs=0.05)
    assert mt._features["b"]._m[0] == pytest.approx(-0.7, abs=0.05)
    # Task 1 (binomial) slopes are in the right direction with
    # plausible magnitudes (less precise because of the logit link).
    assert mt._features["a"]._m[1] > 0.4
    assert mt._features["b"]._m[1] > 0.0

    preds = mt.predict(X)
    assert preds[0].shape == (n,)
    assert preds[1].shape == (n,)
    # Binomial predictions live in (0, 1).
    assert preds[1].min() > 0.0
    assert preds[1].max() < 1.0


def test_predict_accepts_per_task_dataframes() -> None:
    """Passing a list of K DataFrames (each with its own rows) returns
    K predictions of matching length.
    """
    Xs, ys, _ = _gen_two_task_normal(n1=100, n2=160, seed=5)
    mt = MultiTaskGAM(families=["normal", "normal"])
    mt.add_feature("a", type="linear")
    mt.add_feature("b", type="linear")
    mt.fit(Xs, ys, max_its=80)

    Xt1 = pd.DataFrame({"a": np.linspace(-1, 1, 7), "b": np.zeros(7)})
    Xt2 = pd.DataFrame({"a": np.zeros(13), "b": np.linspace(-2, 2, 13)})
    preds = mt.predict([Xt1, Xt2])
    assert preds[0].shape == (7,)
    assert preds[1].shape == (13,)


def test_deviance_matches_predict() -> None:
    """``deviance(X, y)`` evaluated on training data should coincide
    with the deviance computed from the iterate via
    ``deviance()`` to within ADMM tolerance.
    """
    Xs, ys, _ = _gen_two_task_normal(seed=6)
    mt = MultiTaskGAM(families=["normal", "normal"])
    mt.add_feature("a", type="linear")
    mt.add_feature("b", type="linear")
    mt.fit(Xs, ys, max_its=200)

    dev_internal = mt.deviance()
    dev_external = mt.deviance(Xs, ys)
    for k in range(2):
        assert dev_internal[k] == pytest.approx(dev_external[k], rel=5e-3)


def test_predict_before_fit_raises() -> None:
    mt = MultiTaskGAM(families=["normal", "normal"])
    mt.add_feature("a", type="linear")
    X = pd.DataFrame({"a": np.array([0.0, 1.0])})
    with pytest.raises(AttributeError, match="not yet fit"):
        mt.predict([X, X])


def test_summary_and_friends_raise_not_implemented() -> None:
    """Single-task helpers we deliberately did not wire up should surface
    a clear NotImplementedError in this slice.
    """
    mt = MultiTaskGAM(families=["normal", "normal"], name="mt")
    for name in ("summary", "aic", "aicc"):
        with pytest.raises(NotImplementedError):
            getattr(mt, name)()
    with pytest.raises(NotImplementedError):
        mt.gcv()
    with pytest.raises(NotImplementedError):
        mt.ubre()
    with pytest.raises(NotImplementedError):
        mt.confidence_intervals()


def test_unsupported_feature_type_raises() -> None:
    mt = MultiTaskGAM(families=["normal", "normal"])
    with pytest.raises(ValueError, match="not supported in this slice"):
        mt.add_feature("a", type="categorical")  # type: ignore[arg-type]


def test_fit_validates_input_lengths() -> None:
    mt = MultiTaskGAM(families=["normal", "normal"])
    mt.add_feature("a", type="linear")
    X = pd.DataFrame({"a": np.array([0.0, 1.0, 2.0])})
    y = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="must be a list of length 2"):
        mt.fit([X], [y, y], max_its=10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rows"):
        mt.fit([X, X], [y, np.array([0.0, 1.0])], max_its=10)
