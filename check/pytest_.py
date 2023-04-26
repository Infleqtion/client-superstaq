#!/usr/bin/env python3

import sys

import general_superstaq.check

if __name__ == "__main__":
    args = sys.argv[1:]
    args += ["-x", "qiskit-superstaq/examples/uchicago_workshop.ipynb"]
    args += ["-x", "docs/source/apps/*"]
    args += ["-x", "docs/source/get_started/*"]
    args += ["-x", "docs/source/optimizations/aqt/*"]
    exit(general_superstaq.check.pytest_.run(*args))
