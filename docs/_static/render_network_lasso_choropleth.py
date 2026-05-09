"""Render the network-lasso choropleth figure for the user guide.

Run from the repo root with:

    uv run python docs/_static/render_network_lasso_choropleth.py

Writes ``docs/_static/network_lasso_choropleth.png``.

The script is checked in only as a record of how the figure was produced,
so the image stays reproducible if data, geometry, or styling assumptions
change. It is not invoked by the docs build.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from libpysal.examples import load_example
from libpysal.weights import Queen

from gamdist import GAM


def main() -> None:
    ex = load_example("georgia")
    counties = gpd.read_file(ex.get_path("G_utm.shp"))
    counties["county_id"] = counties["AreaKey"].astype(str)

    w = Queen.from_dataframe(counties, use_index=False)
    pairs: list[tuple[str, str]] = []
    for i, neighbors in w.neighbors.items():
        for j in neighbors:
            if i < j:
                pairs.append(
                    (counties.iloc[i]["county_id"], counties.iloc[j]["county_id"])
                )
    edges = pd.DataFrame(pairs, columns=["node1", "node2"])
    edges["weight"] = 1.0

    rng = np.random.default_rng(0)
    atlanta_lat, atlanta_lon = 33.75, -84.39
    dist_to_atl = np.sqrt(
        (counties["Latitude"] - atlanta_lat) ** 2
        + (counties["Longitud"] - atlanta_lon) ** 2
    )
    counties["true_log_income"] = 10.8 - 0.25 * dist_to_atl

    obs_rows: list[dict[str, object]] = []
    for _, row in counties.iterrows():
        for _ in range(5):
            obs_rows.append(
                {
                    "county_id": row["county_id"],
                    "log_income": row["true_log_income"]
                    + rng.normal(0, 0.30),
                }
            )
    obs = pd.DataFrame(obs_rows)
    counties["observed_mean"] = counties["county_id"].map(
        obs.groupby("county_id")["log_income"].mean()
    )

    mdl = GAM(family="normal")
    mdl.add_feature(
        name="county_id",
        type="categorical",
        regularization={"network_lasso": {"coef": 0.1, "edges": edges}},
    )
    mdl.fit(obs[["county_id"]], obs["log_income"].to_numpy())
    counties["fitted_log_income"] = mdl.predict(counties[["county_id"]])

    cols = ["observed_mean", "fitted_log_income", "true_log_income"]
    titles = ["Observed (noisy survey mean)", "Network-lasso fit", "Truth"]
    vmin = counties[cols].min().min()
    vmax = counties[cols].max().max()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    for ax, col, title in zip(axes, cols, titles, strict=True):
        counties.plot(
            column=col,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            edgecolor="white",
            linewidth=0.2,
            ax=ax,
        )
        ax.set_axis_off()
        ax.set_title(title, fontsize=11)

    sm = plt.cm.ScalarMappable(
        cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    cbar = fig.colorbar(
        sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.04, shrink=0.6
    )
    cbar.set_label("log median household income (simulated)")

    fig.suptitle(
        "Spatial smoothing with the network lasso, Georgia counties",
        fontsize=12,
        y=0.98,
    )

    out = Path(__file__).resolve().parent / "network_lasso_choropleth.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
