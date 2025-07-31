Using Supermarq
===============

The benchmarks are defined as classes within :code:`supermarq/benchmarks/`. Each application
defines two methods; :code:`circuit` and :code:`score`. These methods are used to generate the
benchmarking circuit and evaluate its performance
after execution on hardware.

The quantum benchmarks within Supermarq are designed to be scalable, meaning that the benchmarks can be
instantiated and generated for a wide range of circuit sizes and depths.

The notebooks below contain an end-to-end example of how to execute the GHZ benchmark
using `Superstaq <https://superstaq.infleqtion.com/>`_. The general workflow is as follows:

.. code::

    import supermarq

    ghz = supermarq.benchmarks.ghz.GHZ(num_qubits=3)
    ghz_circuit = ghz.circuit()
    counts = execute_circuit_on_quantum_hardware(ghz_circuit) # For example, via AWS Braket, IBM Qiskit, or Superstaq
    score = ghz.score(counts)

.. toctree::
    :maxdepth: 1

    Supermarq_HPCA_Tutorial_css