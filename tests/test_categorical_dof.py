"""Tests for regularization-aware ``_CategoricalFeature.dof()``.

The plain (unregularized) categorical feature reports ``K - 1`` effective
degrees of freedom, where ``K`` is the number of categories. Once a
regularization term is attached, the *fitted* parameter count generally
shrinks: ridge contracts the per-category effects, network lasso fuses
neighbors to identical values, group lasso can zero the feature out, and
so on. These tests pin the expected limits and qualitative behavior of
each regime; they exist to prevent regressions in the AIC / BIC / GCV
chain that consumes ``dof()`` (issue #79).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM
from gamdist.categorical_feature import _CategoricalFeature


def _chain_edges(regions: list[str], weight: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "node1": regions[:-1],
            "node2": regions[1:],
            "weight": [weight] * (len(regions) - 1),
        }
    )


def _balanced_x(regions: list[str], per_region: int) -> np.ndarray:
    return np.repeat(np.array(regions), per_region)


def test_unregularized_dof_is_K_minus_one() -> None:
    feat = _CategoricalFeature(name="g")
    feat.initialize(_balanced_x(["a", "b", "c", "d"], 5))
    feat.optimize(np.zeros(20), rho=1.0)
    assert feat.dof() == pytest.approx(3.0)


def test_l2_dof_shrinks_with_lambda() -> None:
    # Trace formula: edof = sum_c n_c / (n_c + λ_c) - 1.
    # With balanced design (n_c = 10) and uniform λ:
    #   λ → 0     ⇒  edof → 4 - 1 = 3
    #   λ = 10    ⇒  edof = 4 * 10 / (10 + 10) - 1 = 1.0
    #   λ → ∞     ⇒  edof → -1, clamped to 0
    regions = ["a", "b", "c", "d"]
    x = _balanced_x(regions, 10)
    fpumz = np.zeros(40)

    f0 = _CategoricalFeature(name="g", regularization={"l2": {"coef": 1e-9}})
    f0.initialize(x)
    f0.optimize(fpumz, rho=1.0)
    assert f0.dof() == pytest.approx(3.0, abs=1e-3)

    f_mid = _CategoricalFeature(name="g", regularization={"l2": {"coef": 10.0}})
    f_mid.initialize(x)
    f_mid.optimize(fpumz, rho=1.0)
    assert f_mid.dof() == pytest.approx(1.0, abs=1e-6)

    f_inf = _CategoricalFeature(name="g", regularization={"l2": {"coef": 1e12}})
    f_inf.initialize(x)
    f_inf.optimize(fpumz, rho=1.0)
    assert f_inf.dof() == pytest.approx(0.0, abs=1e-6)


def test_l1_dof_counts_active_set() -> None:
    rng = np.random.default_rng(0)
    regions = ["a", "b", "c", "d"]
    x = _balanced_x(regions, 50)
    fpumz = rng.normal(size=200)

    # Lambda small enough to keep everyone active: dof = K - 1.
    f_small = _CategoricalFeature(name="g", regularization={"l1": {"coef": 0.01}})
    f_small.initialize(x)
    f_small.optimize(fpumz, rho=1.0)
    assert f_small.dof() == pytest.approx(3.0)

    # Lambda huge: every coefficient shrinks to 0.
    f_huge = _CategoricalFeature(name="g", regularization={"l1": {"coef": 1e6}})
    f_huge.initialize(x)
    f_huge.optimize(fpumz, rho=1.0)
    assert f_huge.dof() == pytest.approx(0.0, abs=1e-6)


def test_network_lasso_dof_counts_unique_values() -> None:
    # Chain of 8 regions with a smooth signal. Heavy fusion at large λ.
    regions = [f"r{i:02d}" for i in range(8)]
    rng = np.random.default_rng(1)
    n_per = 100
    x = _balanced_x(regions, n_per)
    truth = np.array([float(i) for i in range(8)])
    fpumz = np.repeat(truth, n_per) + rng.normal(scale=0.1, size=8 * n_per)
    edges = _chain_edges(regions)

    # Tiny λ: barely any fusion, dof close to K - 1.
    f_small = _CategoricalFeature(
        name="g",
        regularization={"network_lasso": {"coef": 1e-6, "edges": edges}},
    )
    f_small.initialize(x)
    f_small.optimize(fpumz, rho=1.0)
    assert f_small.dof() == pytest.approx(7.0)

    # Huge λ: fuses everything to a single (zero) value, dof = 0.
    f_huge = _CategoricalFeature(
        name="g",
        regularization={"network_lasso": {"coef": 1e6, "edges": edges}},
    )
    f_huge.initialize(x)
    f_huge.optimize(fpumz, rho=1.0)
    assert f_huge.dof() == pytest.approx(0.0, abs=1e-6)


def test_network_lasso_dof_matches_unique_count_at_intermediate_lambda() -> None:
    rng = np.random.default_rng(2)
    regions = [f"r{i:02d}" for i in range(10)]
    n_per = 60
    x = _balanced_x(regions, n_per)
    truth = np.array([0, 0, 0, 0, 5, 5, 5, 5, 5, 5], dtype=float)
    fpumz = np.repeat(truth, n_per) + rng.normal(scale=0.5, size=10 * n_per)
    edges = _chain_edges(regions)

    feat = _CategoricalFeature(
        name="g",
        regularization={"network_lasso": {"coef": 1.0, "edges": edges}},
    )
    feat.initialize(x)
    feat.optimize(fpumz, rho=1.0)
    n_unique = int(np.unique(np.round(feat.p, 4)).size)
    assert feat.dof() == pytest.approx(float(n_unique - 1), abs=1.0)


def test_network_ridge_dof_shrinks_smoothly_with_lambda() -> None:
    # Network ridge tightens neighbors continuously; dof shrinks but never
    # collapses to 0 unless λ is enormous.
    rng = np.random.default_rng(3)
    regions = [f"r{i:02d}" for i in range(6)]
    n_per = 30
    x = _balanced_x(regions, n_per)
    fpumz = rng.normal(size=6 * n_per)
    edges = _chain_edges(regions)

    f_small = _CategoricalFeature(
        name="g",
        regularization={"network_ridge": {"coef": 1e-9, "edges": edges}},
    )
    f_small.initialize(x)
    f_small.optimize(fpumz, rho=1.0)
    assert f_small.dof() == pytest.approx(5.0, abs=1e-3)

    f_mid = _CategoricalFeature(
        name="g",
        regularization={"network_ridge": {"coef": 5.0, "edges": edges}},
    )
    f_mid.initialize(x)
    f_mid.optimize(fpumz, rho=1.0)
    assert 0.5 < f_mid.dof() < 5.0

    f_huge = _CategoricalFeature(
        name="g",
        regularization={"network_ridge": {"coef": 1e8, "edges": edges}},
    )
    f_huge.initialize(x)
    f_huge.optimize(fpumz, rho=1.0)
    assert f_huge.dof() == pytest.approx(0.0, abs=1e-3)


def test_group_lasso_zeroed_feature_dof_is_zero() -> None:
    rng = np.random.default_rng(4)
    regions = ["a", "b", "c", "d"]
    x = _balanced_x(regions, 40)
    fpumz = rng.normal(scale=0.01, size=160)  # Tiny signal, easy to zero out.

    feat = _CategoricalFeature(name="g", regularization={"group_lasso": {"coef": 1e3}})
    feat.initialize(x)
    feat.optimize(fpumz, rho=1.0)
    # Zeroed out by the heavy group penalty.
    assert np.max(np.abs(feat.p)) < 1e-6
    assert feat.dof() == pytest.approx(0.0)


def test_group_lasso_active_feature_dof_is_K_minus_one() -> None:
    rng = np.random.default_rng(5)
    regions = ["a", "b", "c", "d"]
    x = _balanced_x(regions, 200)
    truth = {"a": 1.0, "b": -1.0, "c": 0.5, "d": -0.5}
    fpumz = np.array([truth[r] for r in x]) + rng.normal(scale=0.05, size=800)

    feat = _CategoricalFeature(name="g", regularization={"group_lasso": {"coef": 0.01}})
    feat.initialize(x)
    feat.optimize(fpumz, rho=1.0)
    assert np.max(np.abs(feat.p)) > 0.1
    assert feat.dof() == pytest.approx(3.0)


def test_full_gam_dispersion_uses_shrunk_dof_for_normal_family() -> None:
    # Before the fix, the divide-by-zero in dispersion() fired whenever
    # n_obs == K (one observation per category). With network_lasso fusing
    # neighbors, dof drops below K and dispersion is well-defined.
    rng = np.random.default_rng(6)
    regions = [f"r{i:02d}" for i in range(20)]
    n_per = 1
    x = _balanced_x(regions, n_per)
    truth = np.array([float(i // 4) for i in range(20)])
    y = truth + rng.normal(scale=0.3, size=20)
    edges = _chain_edges(regions)

    mdl = GAM(family="normal")
    mdl.add_feature(
        name="region",
        type="categorical",
        regularization={"network_lasso": {"coef": 0.5, "edges": edges}},
    )
    mdl.fit(pd.DataFrame({"region": x}), y)

    # The fit should produce some fusion, so dof < n_obs.
    assert mdl.dof() < 20
    # And dispersion should now compute without error.
    phi = mdl.dispersion()
    assert np.isfinite(phi)
    assert phi > 0


def test_huber_dof_matches_ridge_in_l2_limit() -> None:
    # As delta -> infinity Huber collapses to 0.5 * lambda * q^2
    # elementwise (cf. test_huber_large_delta_matches_ridge), i.e. ridge
    # with strength lambda / 2. The corresponding edof must match too:
    # at lambda_huber == 2 * n_c the L2-zone trace formula
    # n_c / (n_c + 0.5 * lambda_h) gives 0.5 per category, so summed over
    # K = 4 balanced categories minus the zero-sum constraint we expect 1.
    regions = ["a", "b", "c", "d"]
    x = _balanced_x(regions, 10)
    fpumz = np.zeros(40)

    feat = _CategoricalFeature(
        name="g", regularization={"huber": {"coef": 20.0, "delta": 1e4}}
    )
    feat.initialize(x)
    feat.optimize(fpumz, rho=1.0)
    assert feat.dof() == pytest.approx(1.0, abs=1e-3)

    # Parity: ridge with half the coef should land on the same edof.
    ridge = _CategoricalFeature(name="g", regularization={"l2": {"coef": 10.0}})
    ridge.initialize(x)
    ridge.optimize(fpumz, rho=1.0)
    assert feat.dof() == pytest.approx(ridge.dof(), abs=1e-6)


def test_dispersion_raises_on_saturated_normal_model() -> None:
    rng = np.random.default_rng(7)
    regions = [f"r{i:02d}" for i in range(10)]
    x = _balanced_x(regions, 1)
    y = rng.normal(size=10)

    mdl = GAM(family="normal")
    mdl.add_feature(name="region", type="categorical")
    mdl.fit(pd.DataFrame({"region": x}), y)

    with pytest.raises(ValueError, match="saturated"):
        mdl.dispersion()
