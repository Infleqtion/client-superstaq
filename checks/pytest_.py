#!/usr/bin/env python3
import sys

import checks_superstaq as checks

if __name__ == "__main__":
    args = sys.argv[1:]
    args += ["--exclude", "docs/source/apps/*"]
    args += ["--exclude", "docs/source/optimizations/ibm/ibmq_dd.ipynb"]
    exit(checks.pytest_.run(*args))
