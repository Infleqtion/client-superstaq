#!/usr/bin/env python3
from __future__ import annotations

import sys

import checks_superstaq as checks

if __name__ == "__main__":
    exit(
        checks.build_docs.run(
            *sys.argv[1:],
            sphinx_paths=[
                "../cirq-superstaq",
                "../qiskit-superstaq",
                "../general-superstaq",
                "../supermarq-benchmarks",
            ],
        )
    )
