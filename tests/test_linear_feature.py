"""Unit tests for the _LinearFeature class."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gamdist.linear_feature import _LinearFeature


def test_requires_name() -> None:
    with pytest.raises(ValueError, match="Feature must have a name"):
        _LinearFeature(name=None)


def test_initialize_centers_data() -> None:
    feat = _LinearFeature(name="x")
    x = np.array([1.0, 2.0, 3.0, 4.0])
    feat.initialize(x)
    assert feat._xmean == pytest.approx(2.5)
    np.testing.assert_allclose(feat._x, x - 2.5)


def test_initialize_with_transform() -> None:
    feat = _LinearFeature(name="x", transform=np.log1p)
    x = np.array([0.0, 1.0, 3.0])
    feat.initialize(x)
    expected_mean = np.log1p(x).mean()
    assert feat._xmean == pytest.approx(expected_mean)


def test_optimize_recovers_slope() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    true_slope = 1.7
    y = true_slope * (x - x.mean())
    feat = _LinearFeature(name="x")
    feat.initialize(x)
    # `fpumz` plays the role of the negation of the residual target.
    feat.optimize(-y, rho=1.0)
    assert feat._m == pytest.approx(true_slope, rel=1e-6)
    assert feat._b == pytest.approx(-true_slope * x.mean(), rel=1e-6)


def test_l2_regularization_shrinks_slope() -> None:
    rng = np.random.default_rng(1)
    n = 50
    x = rng.normal(size=n)
    y = 2.0 * (x - x.mean())

    unreg = _LinearFeature(name="x")
    unreg.initialize(x)
    unreg.optimize(-y, rho=1.0)

    reg = _LinearFeature(name="x", regularization={"l2": {"coef": 5.0}})
    reg.initialize(x)
    reg.optimize(-y, rho=1.0)

    assert abs(reg._m) < abs(unreg._m)


def test_predict_uses_transform() -> None:
    feat = _LinearFeature(name="x", transform=np.log1p)
    x = np.array([0.0, 1.0, 3.0, 7.0])
    feat.initialize(x)
    feat._m = 1.0
    feat._b = 0.0
    np.testing.assert_allclose(feat.predict(x), np.log1p(x))


def test_save_load_round_trip(tmp_path: Path) -> None:
    feat = _LinearFeature(name="x")
    feat.initialize(np.arange(5.0), save_flag=True, save_prefix=str(tmp_path / "model"))
    feat._m = 1.25
    feat._b = -0.5
    feat._save()

    restored = _LinearFeature(load_from_file=feat._filename)
    assert restored._m == pytest.approx(1.25)
    assert restored._b == pytest.approx(-0.5)


def test_dof_and_num_params() -> None:
    feat = _LinearFeature(name="x")
    feat.initialize(np.array([0.0, 1.0]))
    assert feat.num_params() == 1
    assert feat.dof() == 1.0


def test_l1_regularization_rejected() -> None:
    with pytest.raises(ValueError, match="L1 regularization on linear features"):
        _LinearFeature(name="x", regularization={"l1": {"coef": 1.0}})
