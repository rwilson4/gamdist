"""Tests for regularization branches in _CategoricalFeature.initialize."""

from __future__ import annotations

import numpy as np
import pytest

from gamdist.categorical_feature import _CategoricalFeature


def test_l2_regularization_with_float_coef_no_prior() -> None:
    feat = _CategoricalFeature(name="g", regularization={"l2": {"coef": 0.5}})
    feat.initialize(np.array(["a", "b", "a"]))
    np.testing.assert_allclose(feat._lambda2, [0.5, 0.5])


def test_l2_regularization_with_dict_coef_no_prior() -> None:
    feat = _CategoricalFeature(
        name="g", regularization={"l2": {"coef": {"a": 1.0, "b": 0.5}}}
    )
    feat.initialize(np.array(["a", "b", "c"]))
    a = feat._lambda2[feat._category_hash["a"]]
    b = feat._lambda2[feat._category_hash["b"]]
    c = feat._lambda2[feat._category_hash["c"]]
    assert a == pytest.approx(1.0)
    assert b == pytest.approx(0.5)
    assert c == pytest.approx(0.0)


def test_l2_regularization_with_prior() -> None:
    feat = _CategoricalFeature(
        name="g",
        regularization={
            "l2": {"coef": 0.5},
            "prior": {"a": 1.0, "b": -1.0},
        },
    )
    feat.initialize(np.array(["a", "b", "c"]))
    assert feat._prior[feat._category_hash["a"]] == pytest.approx(1.0)
    assert feat._prior[feat._category_hash["b"]] == pytest.approx(-1.0)
    assert feat._prior[feat._category_hash["c"]] == pytest.approx(0.0)
    # Categories not in prior get zero weight on the regularizer.
    c_idx = feat._category_hash["c"]
    assert feat._lambda2[c_idx] == 0.0


def test_l2_no_coef_raises() -> None:
    with pytest.raises(ValueError, match="No coefficient specified for l2"):
        _CategoricalFeature(name="g", regularization={"l2": {}})


def test_l1_no_coef_raises() -> None:
    with pytest.raises(ValueError, match="No coefficient specified for l1"):
        _CategoricalFeature(name="g", regularization={"l1": {}})


def test_smoothing_scales_lambda() -> None:
    feat = _CategoricalFeature(name="g", regularization={"l2": {"coef": 1.0}})
    feat.initialize(np.array(["a", "b"]), smoothing=2.5)
    np.testing.assert_allclose(feat._lambda2, [2.5, 2.5])


def test_initialize_with_covariate_class_sizes() -> None:
    feat = _CategoricalFeature(name="g")
    x = np.array(["a", "b", "a"])
    ccs = np.array([10.0, 20.0, 30.0])
    feat.initialize(x, covariate_class_sizes=ccs)
    a_idx = feat._category_hash["a"]
    b_idx = feat._category_hash["b"]
    assert feat._ccs[a_idx] == pytest.approx(40.0)
    assert feat._ccs[b_idx] == pytest.approx(20.0)
