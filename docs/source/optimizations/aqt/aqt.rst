Superstaq for AQT
========================
In this tutorial, we will go over how to get started using Superstaq to compile and optimize quantum circuits for the Advanced Quantum Testbed (AQT), a superconducting transmon quantum computing testbed at Lawrence Berkeley National Laboratory.

We will discuss how to set up and run Superstaq's AQT compiler with either Qiskit or Cirq, with examples of some of the Superstaq's unique tools and techniques for compiling and optimizing circuits for AQT with different hardware configurations. Some of the strategies demonstrated include:

* Optimized gate decomposition taking advantage of an over-complete set of basis gates.
* Circuit compilation and optimization using your own gate definitions and pulse calibrations.
* Compilation for Equivalent Circuit Averaging (ECA), a strategy for mitigating systematic errors by compiling and running a collection of logically equivalent but physically distinct pulse sequences.

The Superstaq compiler for AQT is designed to be used with ``qtrl``, the control software suite for the Quantum Nanoelectronics Laboratory (QNL) at the University of California, Berkeley. A local installation of ``qtrl`` is required to generate pulse sequences and plots for the examples in this tutorial. Without it, you will still be able to follow along by observing the optimized (Qiskit or Cirq) circuits returned by the AQT compiler in each example.

Below, you will find links to tutorials demonstrate Superstaq for IBM Quantum using either Qiskit or Cirq.

.. toctree::
    :maxdepth: 1

    aqt_qss
    aqt_css

References
-----------------------
To learn more about some of the techniques used in Superstaq compilation for AQT, including a description of ECA, see this paper: Hashim, Akel, et al. "Optimized SWAP networks with equivalent circuit averaging for QAOA". Phys. Rev. Research 4, 033028 (2022).

More information about the Advanced Quantum Testbed can be found at https://aqt.lbl.gov.
