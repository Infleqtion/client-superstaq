#!/usr/bin/env python3

import sys

import general_superstaq.check

if __name__ == "__main__":
    args = sys.argv[1:]
    args += ["-x", "cirq-superstaq/examples/aqt.ipynb"]
    args += ["-x", "qiskit-superstaq/examples/aqt.ipynb"]
    args += ["-x", "qiskit-superstaq/examples/uchicago_workshop.ipynb"]
    args += ["-x", "docs/*"]
    exit(general_superstaq.check.pytest_.run(*args))
