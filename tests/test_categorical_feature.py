"""Unit tests for the _CategoricalFeature class."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gamdist.categorical_feature import _CategoricalFeature


def test_requires_name() -> None:
    with pytest.raises(ValueError, match="Feature must have a name"):
        _CategoricalFeature(name=None)


def test_initialize_assigns_indices() -> None:
    feat = _CategoricalFeature(name="g")
    feat.initialize(np.array(["a", "b", "a", "c"]))
    assert feat._num_categories == 3
    assert set(feat._categories) == {"a", "b", "c"}
    assert feat.num_params() == 3
    assert feat.dof() == 2.0


def test_category_index_returns_tuple() -> None:
    feat = _CategoricalFeature(name="g")
    feat.initialize(np.array(["a", "b", "a"]))
    cidx, csize = feat.category_index(0)
    assert csize == 2
    assert 0 <= cidx < 2


def test_predict_unknown_category_returns_zero() -> None:
    feat = _CategoricalFeature(name="g")
    feat.initialize(np.array(["a", "b"]))
    feat.p = np.array([1.5, -2.0])
    out = feat.predict(np.array(["a", "b", "z"]))
    assert out[2] == 0.0


def test_compute_Az_and_Atz() -> None:
    feat = _CategoricalFeature(name="g")
    feat.initialize(np.array(["a", "b", "a", "c"]))
    z = np.array([1.0, 2.0, 3.0])
    out = feat._compute_Az(z)
    # Each entry is z[category_index_of_observation]
    for i in range(4):
        assert out[i] == z[feat.x[i]]
    # Round-trip through Atz
    sums = feat._compute_Atz(np.ones(4))
    counts = np.bincount(feat.x, minlength=feat._num_categories).astype(float)
    np.testing.assert_allclose(sums, counts)


def test_optimize_preserves_relative_offsets() -> None:
    rng = np.random.default_rng(0)
    cats = np.array(["a", "b", "c"])
    x = rng.choice(cats, size=400)
    true_p = {"a": 0.5, "b": -0.3, "c": 1.1}
    y = np.array([true_p[c] for c in x])

    feat = _CategoricalFeature(name="g")
    feat.initialize(x)
    feat.optimize(-y, rho=1.0)

    # The constraint c.T q == 0 shifts every fitted offset by a constant, so
    # only the differences are recoverable. Pin to category "a" as the
    # reference level.
    a = feat.p[feat._category_hash["a"]]
    b = feat.p[feat._category_hash["b"]]
    c = feat.p[feat._category_hash["c"]]
    assert b - a == pytest.approx(true_p["b"] - true_p["a"], abs=0.05)
    assert c - a == pytest.approx(true_p["c"] - true_p["a"], abs=0.05)


def test_optimize_satisfies_zero_sum_constraint() -> None:
    rng = np.random.default_rng(1)
    x = rng.choice(np.array(["a", "b", "c"]), size=200)
    feat = _CategoricalFeature(name="g")
    feat.initialize(x)
    feat.optimize(-rng.normal(size=200), rho=1.0)
    weighted = float(feat._ccs.dot(feat.p))
    assert abs(weighted) < 1e-4


def test_save_load_round_trip(tmp_path: Path) -> None:
    feat = _CategoricalFeature(name="g")
    feat.initialize(
        np.array(["a", "b", "a"]),
        save_flag=True,
        save_prefix=str(tmp_path / "model"),
    )
    feat.p = np.array([1.0, -1.0])
    feat._save()
    restored = _CategoricalFeature(load_from_file=feat._filename)
    np.testing.assert_allclose(restored.p, feat.p)
    assert restored._categories == feat._categories
