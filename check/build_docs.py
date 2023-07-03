#!/usr/bin/env python3

import sys

import general_superstaq.check

if __name__ == "__main__":
    exit(
        general_superstaq.check.build_docs.run(
            *sys.argv[1:],
            sphinx_paths=[
                "../cirq-superstaq",
                "../qiskit-superstaq",
                "../general-superstaq",
                "../supermarq-benchmarks",
            ]
        )
    )
