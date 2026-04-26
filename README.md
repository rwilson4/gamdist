# gamdist

A modular toolkit for the GLM/GAM zoo — binary, continuous, or count
outcomes; continuous, categorical, or spline-transformed features;
arbitrary regularization (ridge, L1, group lasso, network lasso,
curvature penalties) attached to whichever terms want it. Every model is
a single convex optimization problem.

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

## Install

Requires Python 3.11+. With [uv][uv]:

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

## Development

```bash
uv run pytest                                 # run the test suite (96 tests)
uv run pytest --cov=gamdist                   # with coverage
uv run mypy gamdist                           # type check
uv run ruff check gamdist tests               # lint
```

CI runs all of the above on Python 3.11 and 3.12 (see
`.github/workflows/ci.yml`).

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
