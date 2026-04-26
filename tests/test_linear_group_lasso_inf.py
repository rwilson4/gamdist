"""Tests for the ``group_lasso_inf`` regularization on _LinearFeature.

For a linear feature the per-feature contribution vector is
``f_j = m * (x - mean(x))``. ``||f_j||_inf = |m| * max|x - mean(x)|``,
so the L_inf group-lasso penalty also reduces to a 1-D soft-threshold
on the slope. The threshold scales with ``max|x - mean(x)|`` rather
than ``sqrt(xtx)``, so for the same coefficient it shrinks differently
than the L2 variant.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM
from gamdist.linear_feature import _LinearFeature


def test_group_lasso_inf_no_coef_raises() -> None:
    with pytest.raises(
        ValueError, match="No coefficient specified for group_lasso_inf"
    ):
        _LinearFeature(name="x", regularization={"group_lasso_inf": {}})


def test_group_lasso_inf_smoothing_scales_lambda() -> None:
    feat = _LinearFeature(name="x", regularization={"group_lasso_inf": {"coef": 0.4}})
    feat.initialize(np.array([0.0, 1.0, 2.0, 3.0]), smoothing=2.5)
    assert feat._has_group_lasso_inf
    assert feat._lambda_group_lasso_inf == pytest.approx(1.0)


def test_group_lasso_inf_lambda_zero_matches_unpenalized() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)

    plain = _LinearFeature(name="x")
    plain.initialize(x)
    plain.optimize(fpumz, rho=1.0)

    zero = _LinearFeature(name="x", regularization={"group_lasso_inf": {"coef": 0.0}})
    zero.initialize(x)
    zero.optimize(fpumz, rho=1.0)

    assert zero._m == pytest.approx(plain._m, rel=1e-12, abs=1e-12)


def test_group_lasso_inf_huge_lambda_zeros_slope() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)
    feat = _LinearFeature(name="x", regularization={"group_lasso_inf": {"coef": 1e6}})
    feat.initialize(x)
    feat.optimize(fpumz, rho=1.0)
    assert feat._m == 0.0


def test_group_lasso_inf_soft_threshold_closed_form() -> None:
    # The L_inf group lasso on a linear feature reduces to soft-thresholding
    # the slope with threshold ``coef * max|x - mean(x)| / rho``.
    rng = np.random.default_rng(7)
    n = 50
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)
    rho = 2.0
    coef = 0.3

    feat = _LinearFeature(name="x", regularization={"group_lasso_inf": {"coef": coef}})
    feat.initialize(x)
    feat.optimize(fpumz, rho=rho)

    x_centered = x - x.mean()
    xtx = float(x_centered.dot(x_centered))
    x_inf = float(np.max(np.abs(x_centered)))
    b = float(x_centered.dot(-fpumz))
    threshold = coef * x_inf / rho
    if abs(b) <= threshold:
        expected = 0.0
    else:
        expected = np.sign(b) * (abs(b) - threshold) / xtx
    assert feat._m == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_group_lasso_inf_threshold_differs_from_l2() -> None:
    # On the same data, the L2 and L_inf group-lasso variants produce
    # different effective thresholds (sqrt(xtx) vs max|x - mean(x)|).
    # For an iid-normal feature with n=200, sqrt(xtx) ~ sqrt(n) >> max|x|
    # so the L2 variant shrinks the slope more aggressively at equal coef.
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n)
    y = 2.0 * (x - x.mean()) + 0.1 * rng.normal(size=n)
    coef = 1.5

    l2 = _LinearFeature(name="x", regularization={"group_lasso": {"coef": coef}})
    l2.initialize(x)
    l2.optimize(-y, rho=1.0)

    linf = _LinearFeature(name="x", regularization={"group_lasso_inf": {"coef": coef}})
    linf.initialize(x)
    linf.optimize(-y, rho=1.0)

    x_centered = x - x.mean()
    assert np.sqrt(x_centered.dot(x_centered)) > np.max(np.abs(x_centered))
    # Same sign, different magnitudes; L2 threshold is bigger so it pulls
    # the slope closer to zero.
    assert abs(l2._m) < abs(linf._m)


def test_group_lasso_inf_combines_with_l2_variant() -> None:
    # The two variants stack additively in the soft-threshold.
    rng = np.random.default_rng(3)
    n = 100
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)
    rho = 1.0
    coef_l2 = 0.2
    coef_linf = 0.5

    feat = _LinearFeature(
        name="x",
        regularization={
            "group_lasso": {"coef": coef_l2},
            "group_lasso_inf": {"coef": coef_linf},
        },
    )
    feat.initialize(x)
    feat.optimize(fpumz, rho=rho)

    x_centered = x - x.mean()
    xtx = float(x_centered.dot(x_centered))
    x_inf = float(np.max(np.abs(x_centered)))
    b = float(x_centered.dot(-fpumz))
    threshold = (coef_l2 * np.sqrt(xtx) + coef_linf * x_inf) / rho
    if abs(b) <= threshold:
        expected = 0.0
    else:
        expected = np.sign(b) * (abs(b) - threshold) / xtx
    assert feat._m == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_group_lasso_inf_save_load_round_trip(tmp_path: Path) -> None:
    feat = _LinearFeature(name="x", regularization={"group_lasso_inf": {"coef": 0.7}})
    feat.initialize(
        np.arange(5.0),
        smoothing=2.0,
        save_flag=True,
        save_prefix=str(tmp_path / "model"),
    )
    feat._m = 1.25
    feat._b = -0.5
    feat._save()

    restored = _LinearFeature(load_from_file=feat._filename)
    assert restored._has_group_lasso_inf
    assert restored._coef_group_lasso_inf == pytest.approx(0.7)
    assert restored._lambda_group_lasso_inf == pytest.approx(1.4)
    assert restored._x_inf == pytest.approx(feat._x_inf)
    assert restored._m == pytest.approx(1.25)
    assert restored._b == pytest.approx(-0.5)


def test_group_lasso_inf_within_gam_drops_noise_feature() -> None:
    rng = np.random.default_rng(2)
    n = 400
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = 1.5 * (signal - signal.mean()) + 0.1 * rng.normal(size=n)
    X = pd.DataFrame({"signal": signal, "noise": noise})

    mdl = GAM(family="normal")
    mdl.add_feature(
        name="signal",
        type="linear",
        regularization={"group_lasso_inf": {"coef": 0.8}},
    )
    mdl.add_feature(
        name="noise",
        type="linear",
        regularization={"group_lasso_inf": {"coef": 0.8}},
    )
    mdl.fit(X, y, max_its=60)

    signal_feat = mdl._features["signal"]
    noise_feat = mdl._features["noise"]
    assert abs(signal_feat._m) > 0.5  # type: ignore[attr-defined]
    assert abs(noise_feat._m) < 0.1 * abs(signal_feat._m)  # type: ignore[attr-defined]
