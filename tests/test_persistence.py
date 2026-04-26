"""Save/load round-trip test for GAM models."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from gamdist import GAM


def test_gam_save_load_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "g": rng.choice(["a", "b", "c"], size=n),
        }
    )
    y = (
        2.0 * X["x"].values
        + np.where(X["g"].values == "a", 0.3, np.where(X["g"].values == "b", -0.4, 0.1))
        + rng.normal(size=n) * 0.05
    )

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        mdl = GAM(family="normal", name="round_trip")
        mdl.add_feature(name="x", type="linear")
        mdl.add_feature(name="g", type="categorical")
        mdl.fit(X, y, max_its=40, save_flag=True)

        yhat_before = mdl.predict(X)

        restored = GAM(load_from_file="round_trip_model.pckl")
        yhat_after = restored.predict(X)
        np.testing.assert_allclose(yhat_after, yhat_before, rtol=1e-12, atol=1e-12)
    finally:
        os.chdir(cwd)


def test_network_lasso_edges_persisted(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    regions = [f"r{i:02d}" for i in range(6)]
    edges = pd.DataFrame(
        {"node1": regions[:-1], "node2": regions[1:], "weight": 1.0}
    )
    n = 200
    X = pd.DataFrame({"r": rng.choice(regions, size=n)})
    y = rng.normal(size=n)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        mdl = GAM(family="normal", name="net_round_trip")
        mdl.add_feature(
            name="r",
            type="categorical",
            regularization={"network_lasso": {"coef": 0.5, "edges": edges}},
        )
        mdl.fit(X, y, max_its=20, save_flag=True)

        restored = GAM(load_from_file="net_round_trip_model.pckl")
        restored_feature = restored._features["r"]
        pd.testing.assert_frame_equal(
            restored_feature._edges.reset_index(drop=True),  # type: ignore[attr-defined]
            edges.reset_index(drop=True),
        )
    finally:
        os.chdir(cwd)
