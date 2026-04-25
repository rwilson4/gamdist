# gamdist

Generalized Additive Models fit via the Alternating Direction Method of
Multipliers (ADMM), following the per-feature decomposition of
[Chu, Keshavarz, Boyd][gamadmm].

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
- `confidence_intervals()` and `aicc()` are not yet implemented and
  raise `NotImplementedError`.
- Some non-canonical family/link combinations are non-convex; the
  ADMM iteration is not guaranteed to converge there. In particular,
  gamma + reciprocal can produce non-positive `mu` on small datasets.

[gamadmm]: https://stanford.edu/~boyd/papers/admm/gam.html
[uv]: https://github.com/astral-sh/uv
