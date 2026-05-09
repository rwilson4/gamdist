"""Tests for ``_SplineFeature.dof()`` under group-lasso selection.

The spline feature already shrinks the active parameter count via the
curvature penalty (``self._dof`` is the trace of the smoother set at
fit time). Group-lasso on top of that is feature-selection: when the
penalty is large enough to zero the entire coefficient vector, ``dof``
must drop to 0 to reflect that the feature is no longer in the model
(issue #79).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamdist import GAM


def test_spline_group_lasso_zeroed_dof_is_zero() -> None:
    # Pure-noise signal; very large group-lasso coef forces the spline
    # coefficients to zero.
    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(0.0, 1.0, size=n)
    y = rng.normal(scale=0.01, size=n)

    mdl = GAM(family="normal")
    mdl.add_feature(
        name="x",
        type="spline",
        regularization={"group_lasso": {"coef": 1e3}},
    )
    mdl.fit(pd.DataFrame({"x": x}), y, max_its=40)

    feat = mdl._features["x"]
    # group_lasso penalizes ||N theta||, not theta directly, so theta
    # can be non-tiny while sitting in the null space of N. The
    # right "zeroed" check is on the predicted contribution.
    np.testing.assert_allclose(feat.predict(x), 0.0, atol=1e-4)
    assert feat.dof() == pytest.approx(0.0)


def test_spline_group_lasso_active_dof_matches_curvature_dof() -> None:
    # Strong signal; small group-lasso coef leaves the spline active.
    # In that case dof() should match the curvature-penalty trace
    # (``self._dof`` set at initialize time).
    rng = np.random.default_rng(1)
    n = 400
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    y = np.sin(2.0 * np.pi * x) + 0.05 * rng.normal(size=n)

    mdl = GAM(family="normal")
    mdl.add_feature(
        name="x",
        type="spline",
        regularization={"group_lasso": {"coef": 0.001}},
    )
    mdl.fit(pd.DataFrame({"x": x}), y, max_its=40)

    feat = mdl._features["x"]
    assert float(np.max(np.abs(feat._theta))) > 0.1
    assert feat.dof() == pytest.approx(feat._dof)
