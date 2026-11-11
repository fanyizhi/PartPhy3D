# Contributing

Use Python 3.10-compatible syntax, typed functions and explicit SI units. Keep new
Python code within 88 columns and explain non-obvious mathematics or boundaries.
Preserve immutable source states and version checks during edits.

Run `python -m ruff check .` and `python -m pytest -q` for relevant changes.
`python scripts/edit_parts.py` is the CPU integration demo. Record real commands
and failures; distinguish implemented code, executed experiments and research claims.

Keep data, checkpoints, credentials and private machine settings outside Git.
Do not present material priors as measurements or paper scores as reproduced results.

Before every push, review `git diff --cached --stat` and clean the tracked tree.
Keep one experiment entry point, consolidate duplicate scripts, and retain only
the current figure set with its source data. Dataset images, reference models,
run logs and draft exports stay local. Repair documentation links when removing
files, and run the applicable tests and source-release check before publishing.
