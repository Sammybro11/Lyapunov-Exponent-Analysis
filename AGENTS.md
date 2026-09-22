# AGENTS.md

Guidance for agents working in this repository.

## Project

Chaos analysis of discrete 2D maps (Hénon, Peter de Jong) via Lyapunov exponents, computed with the QR/Gram-Schmidt renormalization method (see README.md for the math). Pure research/analysis code: no package to install, no tests, no linter config.

## Commands

- Environment: **uv** manages everything (`pyproject.toml` + `uv.lock`, Python 3.14). Never use pip/requirements.txt (the README previously referenced one; it does not exist).
- Sync deps: `uv sync`
- Run entry points as modules from the repo root (they import `src.`, `models.`, `utils.` as namespace packages, so plain `python simulations/henon_main.py` fails on imports):
  - `uv run python -m simulations.henon_main` (long: 61x41 parameter grid x 8000 iterations, then 3 GIFs)
  - `uv run python -m simulations.peter_main` (fast)
- Quick smoke check of the core math: `uv run python -m src.Lyapunov`
- Headless runs: prefix with `MPLBACKEND=Agg` (`Henon_Graph` calls `plt.show()`, which blocks).

## Layout

- `src/Lyapunov.py` — core: Lyapunov exponent computation (unbatched numpy + batched torch) for each map.
- `models/Model_Structures.py` — map factory functions (`Create_*_Map`) returning closures. `Create_Henon_Map` returns either a numpy single-point map or a torch batched map depending on `batched=`.
- `simulations/` — runnable entry points, one per map.
- `utils/graphing.py` — plotting: `Henon_Graph` ((a,b) heatmaps), `Map_Simulation` (GIF of point-cloud evolution, works for any map), `Peter_Attractor` (single-trajectory scatter PNG).
- `Plots/` — generated output; paths are hardcoded relative to repo root (`Plots/...`), so always run from the root.

## Conventions

- Naming is non-standard for Python and kept deliberately: functions are `Snake_Case_With_Capitals` (`Create_Henon_Map`, `Lyapunov_Exponent_Peter`), variables often capitalized (`Map`, `Henon_Func`). Match the existing style rather than PEP 8.
- `Henon_Simulation` was renamed to the generic `Map_Simulation`; it takes a `bounds=(x_min, x_max, y_min, y_max)` kwarg (defaults are tuned for Hénon; Peter needs `(-2.5, 2.5, -2.5, 2.5)`).
- Lyapunov functions share a fixed recipe: initial point `[0.2, 0.2]`, `transient=3000`, `n_iterations=5000`, `EPS=1e-10` clamp on QR diagonals. Keep new maps consistent with this.

## Gotchas

- **Divergence handling differs by map.** The Hénon map diverges for many (a,b); the unbatched numpy path guards with `abs(x) > 1e100 -> NaN` inside the map closure, and the graphing code colors NaN cells. The batched torch path does no NaN handling (float32 -> NaNs propagate into results, which is intended). The Peter de Jong map is bounded by construction (sums of sin/cos), so it needs no divergence guard.
- **Sanity identity:** for any 2D map, lambda1 + lambda2 must equal the mean of `ln|det J|` along the orbit (for Hénon this is exactly `ln|b|`). Use this to validate any new map's Jacobian before trusting its exponents.
- `Batched_Lyapunov_Exponent_Henon` auto-selects torch device (MPS > CUDA > CPU) via `get_device()`.
- The correct Peter de Jong map is `y' = sin(c*x) - cos(d*y)`; an earlier version of this repo had the arguments swapped. Don't "fix" it back.
- Dead code was removed deliberately (double pendulum ODE, RK4, hand-rolled Gram-Schmidt); `np.linalg.qr` / `torch.linalg.qr` are used everywhere instead. Don't re-add helpers without a caller.
