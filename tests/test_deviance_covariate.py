"""Regression test for the covariate-class deviance condition fix.

Before the fix, ``GAM.deviance(X, y, covariate_class_sizes=ccs)`` had an
inverted condition that threw away ``ccs`` and silently used ``m = 1``
when covariate-class sizes *were* supplied. This test asserts the
post-fix behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gamdist import GAM


def test_deviance_uses_covariate_class_sizes_when_provided() -> None:
    rng = np.random.default_rng(0)
    n = 25
    X = pd.DataFrame({"g": rng.choice(["a", "b"], size=n)})
    ccs = np.full(n, 50.0)
    y = rng.binomial(50, p=np.where(X["g"].values == "a", 0.4, 0.6)).astype(float)

    mdl = GAM(family="binomial")
    mdl.add_feature(name="g", type="categorical")
    mdl.fit(X, y, covariate_class_sizes=ccs, max_its=40)

    dev_with_ccs = mdl.deviance(X=X, y=y, covariate_class_sizes=ccs)
    dev_without = mdl.deviance(X=X, y=y)

    # When ccs is supplied, the deviance differs from the m=1 case because
    # the (m - y) * log1p(-mu) term changes scale.
    assert dev_with_ccs != dev_without
    assert np.isfinite(dev_with_ccs)
