#!/usr/bin/env python3
import sys

import checks_superstaq as checks

if __name__ == "__main__":
    args = sys.argv[1:]
    args += ["--exclude", "docs/source/apps/*"]
    args += ["--exclude", "docs/source/optimizations/ibm/ibmq_dd.ipynb"]
    args += [
        "--exclude",
        "client-superstaq/cirq-superstaq/cirq_superstaq/hilbert_daily_integration_test.py",
    ]
    args += [
        "--exclude",
        "client-superstaq/qiskit-superstaq/qiskit_superstaq/hilbert_daily_integration_test.py",
    ]
    exit(checks.pytest_.run(*args))
