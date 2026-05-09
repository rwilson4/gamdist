"""Render the residual-diagnostics figure for the evaluating-fit guide.

Run from the repo root with:

    uv run python docs/_static/render_evaluating_fit.py

Writes ``docs/_static/evaluating_fit.png``.

The script is checked in only as a record of how the figure was produced,
so the image stays reproducible if the rent example, residual definitions,
or styling assumptions change. It is not invoked by the docs build.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from gamdist import GAM


def build_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    n = 500

    sqft = rng.uniform(0.5, 3.0, n)
    neighborhood = rng.choice(["downtown", "midtown", "suburbs", "rural"], n)
    walkability = rng.uniform(0, 1, n)

    neighborhood_effect = (
        pd.Series(neighborhood)
        .map({"downtown": 0.5, "midtown": 0.2, "suburbs": -0.1, "rural": -0.4})
        .to_numpy()
    )
    walk_effect = 0.5 * walkability**0.7

    rent = (
        1.5 + sqft + neighborhood_effect + walk_effect + rng.normal(0, 0.1, n)
    )

    X = pd.DataFrame(
        {"sqft": sqft, "neighborhood": neighborhood, "walkability": walkability}
    )
    return X, rent


def main() -> None:
    X, y = build_dataset()

    mdl1 = GAM(family="normal")
    mdl1.add_feature("sqft", type="linear")
    mdl1.fit(X, y)

    mdl2 = GAM(family="normal")
    mdl2.add_feature("sqft", type="linear")
    mdl2.add_feature("neighborhood", type="categorical")
    mdl2.fit(X, y)

    mdl3 = GAM(family="normal")
    mdl3.add_feature("sqft", type="linear")
    mdl3.add_feature("neighborhood", type="categorical")
    mdl3.add_feature("walkability", type="spline")
    mdl3.fit(X, y)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    res1 = mdl1.residuals("deviance")
    levels = ["downtown", "midtown", "suburbs", "rural"]
    positions = {lvl: i for i, lvl in enumerate(levels)}
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(X))
    xs = np.array([positions[n] for n in X["neighborhood"]]) + jitter
    ax = axes[0, 0]
    ax.scatter(xs, res1, s=10, alpha=0.6)
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_xlabel("neighborhood")
    ax.set_ylabel("Deviance residual")
    ax.set_title("mdl1 (sqft only): residuals vs neighborhood")

    res2 = mdl2.residuals("deviance")
    ax = axes[0, 1]
    order = np.argsort(X["walkability"].to_numpy())
    w_sorted = X["walkability"].to_numpy()[order]
    r_sorted = res2[order]
    ax.scatter(w_sorted, r_sorted, s=10, alpha=0.6)
    ax.axhline(0.0, color="grey", linewidth=0.5)
    smoothed = pd.Series(r_sorted).rolling(40, center=True, min_periods=10).mean()
    ax.plot(w_sorted, smoothed, color="C3", linewidth=2, label="rolling mean")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("walkability")
    ax.set_ylabel("Deviance residual")
    ax.set_title("mdl2 (+ neighborhood): residuals vs walkability")

    res3 = mdl3.residuals("deviance")
    mu3 = mdl3.predict(X)
    ax = axes[1, 0]
    ax.scatter(mu3, res3, s=10, alpha=0.6)
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_xlabel("Fitted (mu)")
    ax.set_ylabel("Deviance residual")
    ax.set_title("mdl3 (+ walkability spline): residuals vs fitted")

    ax = axes[1, 1]
    stats.probplot(res3, plot=ax)
    ax.set_title("mdl3: normal Q-Q")
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[0].set_alpha(0.6)

    fig.suptitle(
        "Residual diagnostics across three nested rent models", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = Path(__file__).resolve().parent / "evaluating_fit.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
