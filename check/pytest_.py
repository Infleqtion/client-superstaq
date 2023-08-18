#!/usr/bin/env python3

import sys

import general_superstaq.check

if __name__ == "__main__":
    args = sys.argv[1:]
    args += ["-x", "cirq-superstaq/examples/resource_estimate.ipynb"]
    args += ["-x", "docs/source/apps/*"]
    args += ["-x", "docs/source/get_started/*"]
    args += ["-x", "docs/source/optimizations/ibm/ibmq_compile_css.ipynb"]
    args += ["-x", "qiskit-superstaq/examples/uchicago_workshop.ipynb"]
    exit(general_superstaq.check.pytest_.run(*args))
