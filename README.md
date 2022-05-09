<p align="center">
  <img width="300" src="https://raw.githubusercontent.com/SupertechLabs/SupermarQ/main/static/SupermarQ_Logo.png">
</p>

---
![Continuous Integration](https://github.com/SupertechLabs/SupermarQ/actions/workflows/ci.yml/badge.svg)


# SupermarQ: A Scalable Quantum Benchmark Suite

[SupermarQ](https://arxiv.org/abs/2202.11045) is a suite of application-oriented benchmarks used to measure the performance of quantum computing systems.

## Installation

The SupermarQ package is available via `pip` and can be installed in your current Python environment with the command:

```
pip install supermarq
```

## Using SupermarQ

The benchmarks are defined as classes within `supermarq/benchmarks/`. Each application
defines two methods; `circuit` and `score`. These methods are used to generate the benchmarking circuit and evaluate its performance
after execution on hardware.

The quantum benchmarks within SupermarQ are designed to be scalable, meaning that the benchmarks can be
instantiated and generated for a wide range of circuit sizes and depths.

The [`examples/ghz_example.py`](examples/ghz_example.py) file contains an end-to-end example of how to execute the GHZ benchmark
using [SuperstaQ](https://superstaq.super.tech/). The general workflow is as follows:

```python
import supermarq

ghz = supermarq.benchmarks.ghz.GHZ(num_qubits=3)
ghz_circuit = ghz.circuit()
counts = execute_circuit_on_quantum_hardware(ghz_circuit) # For example, via AWS Braket, IBM Qiskit, or SuperstaQ
score = ghz.score(counts)
```
