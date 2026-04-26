"""Tests for the l1 regularization on _LinearFeature.

For a linear feature ``g(m) = lambda1 * |m|`` reduces the per-feature
primal step to a 1-D soft-threshold on ``b = x_centeredᵀ y`` with
threshold ``lambda1 / rho``. With ridge present, the closed form is
the elastic net: ``m = soft(b, lambda1/rho) / (xtx + 2*lambda2/rho)``.
With group_lasso also present, the L1 and group-lasso thresholds add.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM
from gamdist.linear_feature import _LinearFeature


def test_l1_smoothing_scales_lambda() -> None:
    feat = _LinearFeature(name="x", regularization={"l1": {"coef": 0.4}})
    feat.initialize(np.array([0.0, 1.0, 2.0, 3.0]), smoothing=2.5)
    assert feat._has_l1
    assert feat._lambda1 == pytest.approx(1.0)


def test_l1_lambda_zero_matches_unpenalized() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)

    plain = _LinearFeature(name="x")
    plain.initialize(x)
    plain.optimize(fpumz, rho=1.0)

    zero = _LinearFeature(name="x", regularization={"l1": {"coef": 0.0}})
    zero.initialize(x)
    zero.optimize(fpumz, rho=1.0)

    assert zero._m == pytest.approx(plain._m, rel=1e-12, abs=1e-12)


def test_l1_huge_lambda_zeros_slope() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)
    feat = _LinearFeature(name="x", regularization={"l1": {"coef": 1e6}})
    feat.initialize(x)
    feat.optimize(fpumz, rho=1.0)
    assert feat._m == 0.0


def test_l1_intermediate_lambda_shrinks() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = 2.0 * (x - x.mean()) + 0.1 * rng.normal(size=n)

    unpenalized = _LinearFeature(name="x")
    unpenalized.initialize(x)
    unpenalized.optimize(-y, rho=1.0)

    moderate = _LinearFeature(name="x", regularization={"l1": {"coef": 5.0}})
    moderate.initialize(x)
    moderate.optimize(-y, rho=1.0)

    assert abs(moderate._m) < abs(unpenalized._m)
    assert moderate._m != 0.0


def test_l1_soft_threshold_closed_form() -> None:
    rng = np.random.default_rng(7)
    n = 50
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)
    rho = 2.0
    coef = 0.3

    feat = _LinearFeature(name="x", regularization={"l1": {"coef": coef}})
    feat.initialize(x)
    feat.optimize(fpumz, rho=rho)

    x_centered = x - x.mean()
    xtx = float(x_centered.dot(x_centered))
    b = float(x_centered.dot(-fpumz))
    threshold = coef / rho
    if abs(b) <= threshold:
        expected = 0.0
    else:
        expected = np.sign(b) * (abs(b) - threshold) / xtx
    assert feat._m == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_l1_combined_with_l2_elastic_net_closed_form() -> None:
    rng = np.random.default_rng(11)
    n = 80
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)
    rho = 3.0
    coef1 = 0.6
    coef2 = 1.5

    feat = _LinearFeature(
        name="x",
        regularization={"l1": {"coef": coef1}, "l2": {"coef": coef2}},
    )
    feat.initialize(x)
    feat.optimize(fpumz, rho=rho)

    x_centered = x - x.mean()
    xtx = float(x_centered.dot(x_centered))
    b = float(x_centered.dot(-fpumz))
    threshold = coef1 / rho
    denom = xtx + 2.0 * coef2 / rho
    if abs(b) <= threshold:
        expected = 0.0
    else:
        expected = np.sign(b) * (abs(b) - threshold) / denom
    assert feat._m == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_l1_combined_with_group_lasso_thresholds_add() -> None:
    rng = np.random.default_rng(13)
    n = 60
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)
    rho = 1.5
    coef_l1 = 0.4
    coef_gl = 0.7

    feat = _LinearFeature(
        name="x",
        regularization={
            "l1": {"coef": coef_l1},
            "group_lasso": {"coef": coef_gl},
        },
    )
    feat.initialize(x)
    feat.optimize(fpumz, rho=rho)

    x_centered = x - x.mean()
    xtx = float(x_centered.dot(x_centered))
    b = float(x_centered.dot(-fpumz))
    threshold = coef_l1 / rho + coef_gl * np.sqrt(xtx) / rho
    if abs(b) <= threshold:
        expected = 0.0
    else:
        expected = np.sign(b) * (abs(b) - threshold) / xtx
    assert feat._m == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_l1_save_load_round_trip(tmp_path: Path) -> None:
    feat = _LinearFeature(name="x", regularization={"l1": {"coef": 0.7}})
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
    assert restored._has_l1
    assert restored._coef1 == pytest.approx(0.7)
    assert restored._lambda1 == pytest.approx(1.4)
    assert restored._m == pytest.approx(1.25)
    assert restored._b == pytest.approx(-0.5)


def test_l1_within_gam_drops_noise_feature() -> None:
    # Two linear features. "signal" actually drives y; "noise" does not.
    # With a moderate L1 applied to *both*, noise should shrink to (near)
    # zero while signal retains meaningful slope.
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
        regularization={"l1": {"coef": 0.3}},
    )
    mdl.add_feature(
        name="noise",
        type="linear",
        regularization={"l1": {"coef": 0.3}},
    )
    mdl.fit(X, y, max_its=60)

    signal_feat = mdl._features["signal"]
    noise_feat = mdl._features["noise"]
    assert abs(signal_feat._m) > 0.5  # type: ignore[attr-defined]
    assert abs(noise_feat._m) < 0.1 * abs(signal_feat._m)  # type: ignore[attr-defined]
