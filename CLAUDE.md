# Risikopegel: notes for Claude Code

Read `Risikopegel_PLAN.md` at the start of every session. Its working rules (§0) and "Things not to do" (§10) are binding.

- **No ZeilenJoule dependency.** Don't use the `zj` harness or the models in `~/llm-data-pipelines/zeilenjoule`. Measure timing and energy inside this repo, and download models fresh. Ask first for anything over ~1 GB.
- Commands go through `task`: `task test`, `task test:leakage`, `task lint`, `task report`.
- Python 3.12 with uv. Lint with ruff, type-check with mypy `--strict`. Source is in `src/risikopegel/`, tests are in `tests/`.
- Splits: use only `risikopegel.splits` (see `docs/split_policy.md`). Every new model pipeline needs a causality test and a shuffled-label test.
- Data: `data/` is git-ignored. Every dataset gets an entry in `docs/data.md`. Every library or regulatory fact gets checked and recorded in `docs/versions.md`.
- Numbers in the README or other docs come only from `results/`, through `task report`.
