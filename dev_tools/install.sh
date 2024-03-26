#!/usr/bin/env sh
echo cd $(dirname $0)/../
pip install -e ./checks-superstaq -e ./general-superstaq[dev] -e ./qiskit-superstaq[dev] -e ./cirq-superstaq[dev] -e ./superstaq-benchmarks[dev]
