#!/usr/bin/env sh
cd $(dirname $0)/..
pip install -e ./checks-superstaq -e ./general-superstaq[dev] -e ./qiskit-superstaq[dev] -e ./cirq-superstaq[dev] -e ./supermarq-benchmarks[dev]
