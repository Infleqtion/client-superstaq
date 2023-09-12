#!/usr/bin/env python3

import sys

import general_superstaq.check

if __name__ == "__main__":
    args = sys.argv[1:]
    args += ["--exclude", "docs/source/apps/*"]
    args += ["--exclude", "docs/source/optimizations/ibm/ibmq_dd.ipynb"]
    exit(general_superstaq.check.pytest_.run(*args))
