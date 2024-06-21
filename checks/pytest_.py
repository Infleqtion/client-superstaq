#!/usr/bin/env python3
import re
import sys

import checks_superstaq as checks

if __name__ == "__main__":
    args = sys.argv[1:]

    # Checking if any argument matches the pattern *_integration_test.py
    if "--integration" not in args and any(
        re.match(r".*_integration_test\.py$", arg) for arg in args
    ):
        args.append("--integration")

    args += ["--exclude", "docs/source/apps/*"]
    args += ["--exclude", "docs/source/optimizations/ibm/ibmq_dd.ipynb"]
    exit(checks.pytest_.run(*args))
