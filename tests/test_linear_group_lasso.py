"""Tests for the group_lasso regularization on _LinearFeature.

For a linear feature the per-feature contribution vector is
``f_j = m * (x - mean(x))``. ``||f_j||_2 = |m| * sqrt(xtx)``, so the
group-lasso penalty acts as a 1-D soft-threshold on the slope and can
zero it out entirely once λ is large enough.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM
from gamdist.linear_feature import _LinearFeature


def test_group_lasso_no_coef_raises() -> None:
    with pytest.raises(ValueError, match="No coefficient specified for group_lasso"):
        _LinearFeature(name="x", regularization={"group_lasso": {}})


def test_group_lasso_smoothing_scales_lambda() -> None:
    feat = _LinearFeature(name="x", regularization={"group_lasso": {"coef": 0.4}})
    feat.initialize(np.array([0.0, 1.0, 2.0, 3.0]), smoothing=2.5)
    assert feat._has_group_lasso
    assert feat._lambda_group_lasso == pytest.approx(1.0)


def test_group_lasso_lambda_zero_matches_unpenalized() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)

    plain = _LinearFeature(name="x")
    plain.initialize(x)
    plain.optimize(fpumz, rho=1.0)

    zero = _LinearFeature(name="x", regularization={"group_lasso": {"coef": 0.0}})
    zero.initialize(x)
    zero.optimize(fpumz, rho=1.0)

    assert zero._m == pytest.approx(plain._m, rel=1e-12, abs=1e-12)


def test_group_lasso_huge_lambda_zeros_slope() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)
    feat = _LinearFeature(name="x", regularization={"group_lasso": {"coef": 1e6}})
    feat.initialize(x)
    feat.optimize(fpumz, rho=1.0)
    assert feat._m == 0.0


def test_group_lasso_intermediate_lambda_shrinks() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    # Strong signal so the unpenalized fit isn't tiny.
    y = 2.0 * (x - x.mean()) + 0.1 * rng.normal(size=n)

    unpenalized = _LinearFeature(name="x")
    unpenalized.initialize(x)
    unpenalized.optimize(-y, rho=1.0)

    moderate = _LinearFeature(name="x", regularization={"group_lasso": {"coef": 5.0}})
    moderate.initialize(x)
    moderate.optimize(-y, rho=1.0)

    assert abs(moderate._m) < abs(unpenalized._m)


def test_group_lasso_soft_threshold_closed_form() -> None:
    # Verify the soft-threshold formula directly.
    rng = np.random.default_rng(7)
    n = 50
    x = rng.normal(size=n)
    fpumz = rng.normal(size=n)
    rho = 2.0
    coef = 0.3

    feat = _LinearFeature(name="x", regularization={"group_lasso": {"coef": coef}})
    feat.initialize(x)
    feat.optimize(fpumz, rho=rho)

    x_centered = x - x.mean()
    xtx = float(x_centered.dot(x_centered))
    # On the first call _m starts at 0, so y = -fpumz.
    b = float(x_centered.dot(-fpumz))
    threshold = coef * np.sqrt(xtx) / rho
    if abs(b) <= threshold:
        expected = 0.0
    else:
        expected = np.sign(b) * (abs(b) - threshold) / xtx
    assert feat._m == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_group_lasso_save_load_round_trip(tmp_path: Path) -> None:
    feat = _LinearFeature(name="x", regularization={"group_lasso": {"coef": 0.7}})
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
    assert restored._has_group_lasso
    assert restored._coef_group_lasso == pytest.approx(0.7)
    assert restored._lambda_group_lasso == pytest.approx(1.4)
    assert restored._m == pytest.approx(1.25)
    assert restored._b == pytest.approx(-0.5)


def test_group_lasso_within_gam_drops_noise_feature() -> None:
    # Two linear features. "signal" actually drives y; "noise" does not.
    # With a moderate group lasso applied to *both*, noise should shrink
    # to (near) zero while signal retains meaningful slope.
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
        regularization={"group_lasso": {"coef": 0.3}},
    )
    mdl.add_feature(
        name="noise",
        type="linear",
        regularization={"group_lasso": {"coef": 0.3}},
    )
    mdl.fit(X, y, max_its=60)

    signal_feat = mdl._features["signal"]
    noise_feat = mdl._features["noise"]
    assert abs(signal_feat._m) > 0.5  # type: ignore[attr-defined]
    assert abs(noise_feat._m) < 0.1 * abs(signal_feat._m)  # type: ignore[attr-defined]
