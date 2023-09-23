#!/usr/bin/env python3
import sys

import checks_superstaq as check

if __name__ == "__main__":
    exit(
        check.build_docs.run(
            *sys.argv[1:],
            sphinx_paths=[
                "../cirq-superstaq",
                "../qiskit-superstaq",
                "../general-superstaq",
                "../supermarq-benchmarks",
            ]
        )
    )
