from __future__ import annotations

import sys

import checks_superstaq as checks


def all() -> None:
    checks.all_.run(*sys.argv[1:])


def build_docs() -> None:
    checks.build_docs.run(*sys.argv[1:])


def configs() -> None:
    checks.configs.run(*sys.argv[1:])


def coverage() -> None:
    checks.coverage_.run(*sys.argv[1:])


def flake8() -> None:
    checks.flake8_.run(*sys.argv[1:])


def format() -> None:
    checks.format_.run(*sys.argv[1:])


def mypy() -> None:
    checks.mypy_.run(*sys.argv[1:])


def pylint() -> None:
    checks.pylint_.run(*sys.argv[1:])


def pytest() -> None:
    args = sys.argv[1:]
    args += ["--exclude", "docs/source/apps/aces/*"]
    args += ["--exclude", "docs/source/apps/dfe/*"]
    args += ["--exclude", "docs/source/apps/supermarq/examples/qre-challenge/*"]
    args += ["--exclude", "docs/source/apps/max_sharpe_ratio_optimization.ipynb"]
    args += ["--exclude", "docs/source/apps/cudaq_logical_aim.ipynb"]
    args += ["--exclude", "docs/source/optimizations/ibm/ibmq_dd.ipynb"]
    checks.pytest_.run(*args)


def requirements() -> None:
    checks.requirements.run(*sys.argv[1:])


def ruff_format() -> None:
    checks.ruff_format_.run(*sys.argv[1:])


def ruff_lint() -> None:
    checks.ruff_lint_.run(*sys.argv[1:])
