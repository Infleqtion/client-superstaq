Superstaq for AQT
========================
In this tutorial, we will go over how to get started using Superstaq to compile and optimize quantum circuits for the Advanced Quantum Testbed (AQT), a superconducting transmon quantum computing testbed at Lawrence Berkeley National Laboratory.

The Superstaq compiler for AQT is designed to be used with `qtrl`, the control software suite for the Quantum Nanoelectronics Laboratory (QNL) at the University of California, Berkeley. A local installation of `qtrl` is required to generate pulse sequences and plots for the examples in this tutorial. Without it, you will still be able to follow along by observing the optimized (Cirq or Qiskit) circuits returned by the AQT compiler in each example.

    "To generate the plots, you'll also need a local installation of `qtrl`, the control software suite for the Qantum Nanoelectronics Laboratory (QNL) at the University of California, Berkeley."

We will discuss how to set up and run the AQT compiler with either Cirq or Qiskit, with
examples of some of the tools provided by Superstaq for customizing and optimizing the circuits for 
With examples of some of the tools provided by Superstaq for optimizing circuits given different

quantum circuit optimizations taking advantage of an over-complete gateset

optimization for compilation to over-complete gatesets

demonstrate some of the tools provided by Superstaq  and techniques provided by Superstaq to compile and optimize circuits  to generate the resulting pulse sequences.

with your own configuration, and provide

We will also discuss Equivalent Circuit Averaging, a strategy for mitigating systematic errors by
compiling and running a collection of logically equivalent but physically distinct pulse sequences.

        """Compiles and optimizes the given circuit(s) for the Advanced Quantum Testbed (AQT).

        AQT is a superconducting transmon quantum computing testbed at Lawrence Berkeley National
        Laboratory. More information can be found at https://aqt.lbl.gov.

        Specifying a nonzero value for `num_eca_circuits` enables compilation with Equivalent
        Circuit Averaging (ECA). See https://arxiv.org/abs/2111.04572 for a description of ECA.


Quantum backends. We will go over the process of setting up a Superstaq provider and compiling circuits to an IBM Quantum backend, retrieving the optimized circuits. 


Below, you will find links to tutorials demonstrate Superstaq for IBM Quantum using either Qiskit or Cirq.

.. toctree::
    :maxdepth: 1

    aqt_qss
    aqt_css

Superstaq for IBM
========================
For this tutorial, we will go over how to get started with using Superstaq to compile, optimize, and simulate circuits for the IBM Quantum backends. We will go over the process of setting up a Superstaq provider and compiling circuits to an IBM Quantum backend, retrieving the optimized circuits. 

This tutorial will also show you how to utilize the various features offered by Superstaq, such as generating compiled circuit Qiskit Pulse schedules. By the end, we will provide a step-by-step guide to simulate a given circuit for an IBM Quantum backend and get back the results from the simulation!

Below, you will find links to identical tutorials that demonstrate Superstaq for IBM Quantum using either Qiskit or Cirq.

.. toctree::
    :maxdepth: 1

    ibmq_compile_qss
    ibmq_compile_css

References
-----------------------
To learn more about some of the techniques used in Superstaq compilation for AQT, including a description of ECA, see this paper: Hashim, Akel, et al. "Optimized SWAP networks with equivalent circuit averaging for QAOA". Phys. Rev. Research 4, 033028 (2022).

More information about the Advanced Quantum Testbed can be found at https://aqt.lbl.gov.
