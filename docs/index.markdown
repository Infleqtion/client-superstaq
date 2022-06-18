---
layout: default
title: "ISCA '22 Tutorial: Scalable Quantum Benchmarking"
---

{:refdef: style="text-align: center;"}
![Logo](/static/SupermarQ_Logo.png){:width="50%"; refdef}

The goal of this tutorial is to:
1. Provide an overview of the current quantum benchmarking landscape.
2. Introduce [SupermarQ](https://arxiv.org/abs/2202.11045), a new quantum benchmarking suite recently published in [HPCA 2022](https://hpca-conf.org/2022/program/#session4c).
3. Walk attendees through a hands-on demo of generating benchmarks, evaluating them on hardware, and processing the results.
4. Show users how they can develop and design their own quantum benchmarks.

# SupermarQ
SupermarQ is a scalable, application-oriented, quantum benchmark suite. It is based on collaborative research between Princeton University, The University of Chicago, Super.tech, and Northwestern University. This work was recently published in [HPCA 2022](https://hpca-conf.org/2022/program/#session4c) and an open source version is available on [arXiv](https://arxiv.org/abs/2202.11045). More information on SupermarQ can be found on the [Super.tech website](https://www.super.tech/supermarq/), and our [GitHub repository](https://github.com/SupertechLabs/SupermarQ).

# Benchmarking Demo

Before getting started (and ideally before attending the ISCA tutorial) please set up a fresh python environment
we can use to run the SupermarQ benchmarks.

Please note that Python version `3.7` or higher **is required**. Once you have the correct version of Python installed,
a new virtual environment can be created via the command:

```
python3 -m venv name_of_your_environment
```

Activate that new environment by

```
source name_of_your_environment/bin/activate
```

Once this is completed you can type the command `which python` to see that you are now referencing the python installed at the location
of your new environment. With your newly created and activated python environment it is time to install all of the necessary packages.

```
pip install supermarq
pip install jupyterlab
```

This will install `supermarq` and its dependencies: `cirq-superstaq` and `qiskit-superstaq` -- which we will use to access different quantum computers available over the cloud.

Start your jupyter lab by running the command:

```
jupyter lab
```

# Location: Chelsea (lower level)

# Schedule (Sunday, June 19, NYC)

- **8:30 - 9:00**: Morning coffee/breakfast
- **9:00 - 9:45**: Overview of quantum benchmarking and in-depth walkthrough of SupermarQ
- **9:45 - 10:00**: Questions and discussion
- **10:00 - 10:30**: Hands-on development -- participants will have the opportunity to implement, profile, and run their own benchmarks on real quantum hardware
- **10:30 - 11:00**: Coffee break
- **11:00 - 12:00** Continued hands-on benchmark development, general questions and discussion
