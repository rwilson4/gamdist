"""Tests for the network_ridge regularization on _CategoricalFeature.

Network ridge applies a quadratic penalty to the differences between
coefficients of categories connected by a user-supplied edge list. The
penalty term is ``lambda * sum_{(i,j) in E} w_ij * (q_i - q_j)^2 =
lambda * q^T L q`` where ``L`` is the (weighted) graph Laplacian. It is
the smooth-shrinkage sibling of ``network_lasso``, which clusters
neighbors to identical coefficients via L1 on the same edge differences.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

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


def test_network_ridge_no_coef_raises() -> None:
    edges = _chain_edges(["a", "b", "c"])
    with pytest.raises(ValueError, match="No coefficient specified for Network Ridge"):
        _CategoricalFeature(
            name="g", regularization={"network_ridge": {"edges": edges}}
        )


def test_network_ridge_no_edges_raises() -> None:
    with pytest.raises(ValueError, match="Edges not specified for Network Ridge"):
        _CategoricalFeature(name="g", regularization={"network_ridge": {"coef": 0.5}})


def test_network_ridge_smoothing_scales_lambda() -> None:
    edges = _chain_edges(["a", "b"])
    feat = _CategoricalFeature(
        name="g", regularization={"network_ridge": {"coef": 0.4, "edges": edges}}
    )
    feat.initialize(np.array(["a", "b", "a"]), smoothing=2.5)
    assert feat._has_network_ridge
    assert feat._lambda_network_ridge == pytest.approx(1.0)


def test_network_ridge_laplacian_structure() -> None:
    # Two-node, single-edge graph: L = [[w, -w], [-w, w]].
    edges = _chain_edges(["a", "b"], weight=2.0)
    feat = _CategoricalFeature(
        name="g", regularization={"network_ridge": {"coef": 1.0, "edges": edges}}
    )
    feat.initialize(np.array(["a", "b"]))
    a = feat._category_hash["a"]
    b = feat._category_hash["b"]
    L = feat._L.toarray()
    assert L[a, a] == pytest.approx(2.0)
    assert L[b, b] == pytest.approx(2.0)
    assert L[a, b] == pytest.approx(-2.0)
    assert L[b, a] == pytest.approx(-2.0)
    # Symmetric and PSD: row sums to zero, eigenvalues all >= 0.
    np.testing.assert_allclose(L.sum(axis=1), 0.0)
    eigvals = np.linalg.eigvalsh(L)
    assert eigvals.min() >= -1e-12


def test_network_ridge_lambda_zero_matches_unpenalized() -> None:
    rng = np.random.default_rng(0)
    x = rng.choice(np.array(["a", "b", "c"]), size=200)
    fpumz = rng.normal(size=200)
    edges = _chain_edges(["a", "b", "c"])

    plain = _CategoricalFeature(name="g")
    plain.initialize(x)
    plain.optimize(fpumz, rho=1.0)

    zero = _CategoricalFeature(
        name="g", regularization={"network_ridge": {"coef": 0.0, "edges": edges}}
    )
    zero.initialize(x)
    zero.optimize(fpumz, rho=1.0)

    np.testing.assert_allclose(zero.p, plain.p, atol=1e-6)


def test_network_ridge_empty_edges_matches_unpenalized() -> None:
    # Empty edges DataFrame: Laplacian is the zero matrix, so the fit
    # should match the unregularized solution.
    rng = np.random.default_rng(1)
    x = rng.choice(np.array(["a", "b", "c"]), size=200)
    fpumz = rng.normal(size=200)
    empty_edges = pd.DataFrame({"node1": [], "node2": [], "weight": []})

    plain = _CategoricalFeature(name="g")
    plain.initialize(x)
    plain.optimize(fpumz, rho=1.0)

    empty = _CategoricalFeature(
        name="g",
        regularization={"network_ridge": {"coef": 5.0, "edges": empty_edges}},
    )
    empty.initialize(x)
    empty.optimize(fpumz, rho=1.0)

    np.testing.assert_allclose(empty.p, plain.p, atol=1e-6)


def test_network_ridge_huge_lambda_collapses_to_constant() -> None:
    # As lambda -> infinity, neighbor differences go to zero. With a
    # connected graph and the zero-sum constraint imposed by ccs, every
    # coefficient must approach 0.
    rng = np.random.default_rng(2)
    x = rng.choice(np.array(["a", "b", "c", "d"]), size=300)
    fpumz = rng.normal(size=300) + 1.0
    edges = _chain_edges(["a", "b", "c", "d"])
    feat = _CategoricalFeature(
        name="g", regularization={"network_ridge": {"coef": 1e6, "edges": edges}}
    )
    feat.initialize(x)
    feat.optimize(fpumz, rho=1.0)
    np.testing.assert_allclose(feat.p, 0.0, atol=1e-3)


def test_network_ridge_intermediate_lambda_shrinks_neighbor_gap() -> None:
    # On a smooth-along-the-chain signal, network ridge should pull
    # neighboring coefficients closer together than the unpenalized fit.
    rng = np.random.default_rng(3)
    regions = [f"r{i:02d}" for i in range(8)]
    true_effect = dict(zip(regions, np.linspace(-1.0, 1.0, len(regions)), strict=True))
    n = 400
    x = rng.choice(np.array(regions), size=n)
    fpumz = np.array([true_effect[r] for r in x]) + rng.normal(scale=0.5, size=n)
    edges = _chain_edges(regions)

    unpenalized = _CategoricalFeature(name="g")
    unpenalized.initialize(x)
    unpenalized.optimize(fpumz, rho=1.0)

    moderate = _CategoricalFeature(
        name="g", regularization={"network_ridge": {"coef": 50.0, "edges": edges}}
    )
    moderate.initialize(x)
    moderate.optimize(fpumz, rho=1.0)

    # Sum of squared neighbor differences should drop substantially.
    def neighbor_ssq(p: np.ndarray, feat: _CategoricalFeature) -> float:
        total = 0.0
        for r1, r2 in pairwise(regions):
            i = feat._category_hash[r1]
            j = feat._category_hash[r2]
            total += float((p[i] - p[j]) ** 2)
        return total

    assert neighbor_ssq(moderate.p, moderate) < 0.5 * neighbor_ssq(
        unpenalized.p, unpenalized
    )


def test_network_ridge_save_load_round_trip(tmp_path: Path) -> None:
    edges = _chain_edges(["a", "b", "c"])
    feat = _CategoricalFeature(
        name="g", regularization={"network_ridge": {"coef": 0.7, "edges": edges}}
    )
    feat.initialize(
        np.array(["a", "b", "a", "c"]),
        smoothing=2.0,
        save_flag=True,
        save_prefix=str(tmp_path / "model"),
    )
    feat.p = np.array([0.3, -0.1, -0.2])
    feat._save()

    restored = _CategoricalFeature(load_from_file=feat._filename)
    assert restored._has_network_ridge
    assert restored._lambda_network_ridge == pytest.approx(1.4)
    assert restored._num_edges_ridge == 2
    np.testing.assert_allclose(restored._L.toarray(), feat._L.toarray())
    pd.testing.assert_frame_equal(
        restored._edges_ridge.reset_index(drop=True),
        edges.reset_index(drop=True),
    )
    np.testing.assert_allclose(restored.p, feat.p)


def test_network_ridge_beats_plain_l2_on_smooth_spatial_signal() -> None:
    # Synthetic spatial setup: 12 regions in a chain with a smooth-along-
    # the-chain true effect. The middle region is undersampled (only a
    # handful of noisy observations), so plain L2 has almost no signal
    # for it and shrinks its coefficient toward zero. Network ridge, by
    # tying the coefficient to its neighbors, should land closer to the
    # true effect at the under-sampled region.
    rng = np.random.default_rng(4)
    regions = [f"r{i:02d}" for i in range(12)]
    centered = np.linspace(-1.0, 1.0, len(regions))
    centered = centered - centered.mean()
    true_effect = dict(zip(regions, centered, strict=True))

    # Pick a region whose true effect is far from zero, so that L2
    # shrinkage toward 0 noticeably misses while neighbor-borrowing
    # via the chain stays close to truth.
    target_region = "r10"
    n_well_sampled = 200
    n_target = 3
    well_sampled_regions = [r for r in regions if r != target_region]
    x_well = rng.choice(np.array(well_sampled_regions), size=n_well_sampled)
    x_target = np.array([target_region] * n_target)
    x_train = np.concatenate([x_well, x_target])
    y_train = np.array([true_effect[r] for r in x_train]) + rng.normal(
        scale=0.3, size=len(x_train)
    )

    X_train = pd.DataFrame({"region": x_train})
    edges = _chain_edges(regions)

    ridge_mdl = GAM(family="normal")
    ridge_mdl.add_feature(
        name="region",
        type="categorical",
        regularization={"network_ridge": {"coef": 50.0, "edges": edges}},
    )
    ridge_mdl.fit(X_train, y_train, max_its=80)

    l2_mdl = GAM(family="normal")
    l2_mdl.add_feature(
        name="region",
        type="categorical",
        regularization={"l2": {"coef": 50.0}},
    )
    l2_mdl.fit(X_train, y_train, max_its=80)

    target_df = pd.DataFrame({"region": [target_region]})
    ridge_pred = ridge_mdl.predict(target_df)[0]
    l2_pred = l2_mdl.predict(target_df)[0]
    target = true_effect[target_region]

    # Network ridge propagates the effect across the adjacency and
    # should land closer to the true value than plain L2, which only
    # has the few noisy direct observations to work with.
    assert abs(ridge_pred - target) < abs(l2_pred - target)


def test_network_ridge_combined_with_l2() -> None:
    # Both penalties active: still solves and produces finite coefficients.
    rng = np.random.default_rng(5)
    x = rng.choice(np.array(["a", "b", "c"]), size=200)
    fpumz = rng.normal(size=200)
    edges = _chain_edges(["a", "b", "c"])
    feat = _CategoricalFeature(
        name="g",
        regularization={
            "l2": {"coef": 0.2},
            "network_ridge": {"coef": 0.5, "edges": edges},
        },
    )
    feat.initialize(x)
    feat.optimize(fpumz, rho=1.0)
    assert np.all(np.isfinite(feat.p))
    # Zero-sum constraint still respected.
    assert abs(feat._ccs @ feat.p) < 1e-6
