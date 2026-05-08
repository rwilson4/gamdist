# gamdist

A modular toolkit for the GLM/GAM zoo — binary, continuous, or count
outcomes; continuous, categorical, or spline-transformed features;
arbitrary regularization (ridge, L1, group lasso, network lasso,
network ridge, curvature penalties) attached to whichever terms want
it. Every model is a single convex optimization problem.

The joint Hessian for that problem is intractable to derive and solve as
a monolith. The ADMM decomposition of [Chu, Keshavarz, Boyd][gamadmm]
makes the zoo tractable by splitting the problem into a per-feature
primal step plus a per-outcome proximal step coordinated by dual
variables. Parallelism is a side effect; the real prize is
**modularity** — outcomes, features, and regularizers are independent
components that mix and match in any combination, with new ones plugging
in without disturbing the rest.

Supported families: `normal`, `binomial`, `poisson`, `gamma`,
`exponential` (= gamma with dispersion 1), `inverse_gaussian`.
Supported links: `identity`, `logistic`, `probit`,
`complementary_log_log`, `log`, `reciprocal`, `reciprocal_squared`.

## Feature types and regularization

Three feature types are available, each with its own set of penalties
applied inside the per-feature ADMM step:

- **`linear`** — continuous feature with a single coefficient. Supports
  ridge (`l2`).
- **`categorical`** — per-level offset for a categorical feature.
  Supports `l1`, `l2`, group lasso (`group_lasso`) for variable
  selection, **network lasso** (`network_lasso`) for clustering
  connected categories to identical coefficients, and **network
  ridge** (`network_ridge`) for smoothly shrinking connected
  categories toward each other.
- **`spline`** — cubic regression spline with an integrated curvature
  penalty. Smoothing is set via `rel_dof`, the target effective degrees
  of freedom.

The network lasso is a good illustration of why the modular design
matters. Pass an `edges` DataFrame describing which categories should
have similar coefficients (neighboring counties, related products,
friends in a social graph), and the categorical feature's optimization
step adds an L1 penalty on the edge differences. No other component of
the model needs to change.

## Install

Requires Python 3.11+.

```bash
pip install gamdist-py
```

For development, clone the repo and use [uv][uv]:

```bash
uv sync                  # runtime deps
uv sync --extra dev      # plus pytest, mypy, ruff
```

## Quickstart

```python
import numpy as np
import pandas as pd
from gamdist import GAM

X = pd.DataFrame(
    {
        "purchases": np.random.choice([0, 3, 10, 16], size=1000),
        "gender": np.random.choice(["male", "female"], size=1000),
    }
)
y = (
    0.1 * np.log1p(X["purchases"].values)
    + np.where(X["gender"].values == "male", 0.1, -0.5)
    + np.random.normal(size=1000) * 0.1
)

mdl = GAM(family="normal")
mdl.add_feature(name="purchases", type="linear", transform=np.log1p)
mdl.add_feature(name="gender", type="categorical")
mdl.fit(X, y)

mdl.summary()
yhat = mdl.predict(X)
```

### Network lasso on a spatial categorical

A second example showing the modular regularization story: 12 regions
arranged in a chain, with a true effect that drifts smoothly along the
chain. The network lasso shrinks neighboring regions toward identical
coefficients without any change to the rest of the model.

```python
import numpy as np
import pandas as pd
from gamdist import GAM

regions = [f"r{i:02d}" for i in range(12)]
true_effect = dict(zip(regions, np.linspace(-1.0, 1.0, len(regions))))

edges = pd.DataFrame(
    {"node1": regions[:-1], "node2": regions[1:], "weight": 1.0}
)

n = 2000
X = pd.DataFrame({"region": np.random.choice(regions, size=n)})
y = (
    np.array([true_effect[r] for r in X["region"]])
    + np.random.normal(scale=0.3, size=n)
)

mdl = GAM(family="normal")
mdl.add_feature(
    name="region",
    type="categorical",
    regularization={"network_lasso": {"coef": 1.0, "edges": edges}},
)
mdl.fit(X, y)
mdl.summary()
```

Swap `network_lasso` for `network_ridge` on the same edges DataFrame
to get the smooth-shrinkage variant: a quadratic penalty
`λ · Σ w_ij · (β_i − β_j)²` (= `λ · βᵀ L β` for the graph Laplacian
`L`) that pulls neighboring coefficients *toward* each other instead
of clustering them to identical values.

### Multi-task: one feature set, multiple outcomes

`MultiTaskGAM` fits K outcomes jointly under a shared feature set, with
each task picking its own family and link and an optional **coupling
regularizer** that ties the K coefficients of each feature together.
The headline use case: identify a feature set that's simultaneously
predictive of every outcome — not a feature set that's brilliant on
some outcomes and useless on others.

```python
import numpy as np
import pandas as pd
from gamdist import MultiTaskGAM

rng = np.random.default_rng(1)
n = 400
signal = rng.normal(size=n)
noise = rng.normal(size=n)
X = pd.DataFrame({"signal": signal, "noise": noise})

y_continuous = 1.5 * signal + 0.1 * rng.normal(size=n)
y_binary = (
    rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.8 * signal))
).astype(float)

mdl = MultiTaskGAM(families=["normal", "binomial"])
mdl.add_feature(
    "signal",
    type="linear",
    regularization={"group_lasso_across_tasks": {"coef": 0.5}},
)
mdl.add_feature(
    "noise",
    type="linear",
    regularization={"group_lasso_across_tasks": {"coef": 0.5}},
)
mdl.fit([X, X], [y_continuous, y_binary])

yhats = mdl.predict([X, X])
# yhats[0] is the continuous task's mean; yhats[1] is task 2's
# probability in (0, 1). With this λ the `noise` feature has been
# dropped from BOTH tasks simultaneously — group-lasso across tasks
# zeros the entire K-vector of slopes at once, not each task's slope
# independently. That's the multi-task variable-selection story you
# can't get from running K independent GAMs.
```

Tasks may have different `(family, link)` pairs, different observation
counts, even different feature data per task — they only have to agree
on the *names* of the features they share. The per-task ADMM splits
stay independent except for the cross-task penalty inside each shared
feature, so the seam principle (CLAUDE.md) carries over without
changes to the orchestrator.

Other coupling penalties (trace / nuclear norm, network-on-tasks,
hierarchical pooling) and other convex ways to combine the per-task
losses (weighted sum, log-sum-exp, minimax / CVaR) are tracked as
follow-ups in [issue #39][issue39] and [issue #71][issue71].

## Development

```bash
uv run pytest                                 # run the test suite (96 tests)
uv run pytest --cov=gamdist                   # with coverage
uv run mypy gamdist                           # type check
uv run ruff check gamdist tests               # lint
```

CI runs all of the above on Python 3.11 and 3.12 (see
`.github/workflows/ci.yml`).

## Extending gamdist

The modular design means new components plug in along well-defined
seams without touching the rest of the system:

- **New outcome distribution / link** — add a proximal operator entry
  in `gamdist/proximal_operators.py` for the `(family, link)` pair.
  Nothing on the feature side changes.
- **New feature type** — subclass `_Feature` (see `gamdist/feature.py`)
  and implement the standard interface (`initialize`, `optimize`,
  `compute_dual_tol`, `num_params`, `dof`, `predict`, `_save`,
  `_load`). The ADMM loop in `gamdist.py` doesn't need to know.
- **New regularizer** — add it inside a feature's `optimize` step
  (alongside the existing L1 / L2 / group-lasso / network-lasso /
  curvature terms), scaled by `smoothing` in `initialize`. The global
  loop never sees a penalty coefficient.

Every per-component subproblem must be convex.

## Caveats

- The package uses `pickle` for save/load and is **not designed for
  untrusted input**.
- `confidence_intervals()` is not yet implemented and raises
  `NotImplementedError`.
- Convexity of every per-component subproblem is a hard requirement.
  Non-convex (family, link) combinations are out of scope; the existing
  `scipy.optimize.minimize_scalar` fallback for non-canonical pairs
  predates this principle and is scheduled for removal (see
  [issue #19][issue19]).
- Gamma + reciprocal can produce non-positive `mu` on small datasets
  (numerical edge case in an otherwise supported combination).

[gamadmm]: https://stanford.edu/~boyd/papers/admm/gam.html
[uv]: https://github.com/astral-sh/uv
[issue19]: https://github.com/rwilson4/gamdist/issues/19
[issue39]: https://github.com/rwilson4/gamdist/issues/39
[issue71]: https://github.com/rwilson4/gamdist/issues/71
