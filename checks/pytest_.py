#!/usr/bin/env python3
from __future__ import annotations

import sys

import checks_superstaq as checks

# Just a random comment.
if __name__ == "__main__":
    args = sys.argv[1:]
    args += ["--exclude", "docs/source/apps/aces/*"]
    args += ["--exclude", "docs/source/apps/dfe/*"]
    args += ["--exclude", "docs/source/apps/supermarq/examples/qre-challenge/*"]
    args += ["--exclude", "docs/source/apps/max_sharpe_ratio_optimization.ipynb"]
    args += ["--exclude", "docs/source/apps/cudaq_logical_aim.ipynb"]
    args += ["--exclude", "docs/source/optimizations/ibm/ibmq_dd.ipynb"]
    sys.exit(checks.pytest_.run(*args))
