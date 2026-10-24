# Contributing

Maintainer: **Olegequ**.

Create a virtual environment and install `python -m pip install -e ".[dev]"`.
Run `python -m pytest`, `python scripts/check_release.py`, and
`airwake-rotor demo --output runs/demo` before proposing a change.

For numerical changes, explain the physical assumption, units, boundary
conditions, and effect on reproducibility. Include a focused regression test
when a changed algorithm or input contract warrants one. Preserve shared trim,
case semantics, nonorthogonal energy accounting, and hard domain boundaries.

Keep real CFD data, generated results, private absolute paths, access tokens,
and unpublished manuscript material out of pull requests. Use small synthetic
fixtures. Report the Python and dependency versions needed to reproduce an
issue. Propose changes through a branch and pull request in the eventual
hosting repository.

Contributions are intended to be distributed under the project's MIT license;
retain applicable third-party notices.
