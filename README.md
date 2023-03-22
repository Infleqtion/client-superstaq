<img src="./docs/source/_static/logos/SuperstaQ_SSa-R00a_Mil.png">

# Welcome to SuperstaQ!
This repository is the home of Super.tech's open source work, which includes:
* [SuperstaQ](https://www.super.tech/about-superstaq/), our quantum software platform that is optimized across the quantum stack and enables users to write quantum programs in Cirq or Qiskit and target a variety of quantum computers and simulators
* [SupermarQ](https://www.super.tech/supermarq/), our quantum benchmarking suite

# Installation for users
For installation instructions for users of SuperstaQ, check out [our documentation site](https://docs-superstaq.readthedocs.io/)! In short, you can install any of our packages by doing `pip install <package>` in a terminal, where `<package>` is `qiskit-superstaq`, `cirq-superstaq`, `general-superstaq`, or `supermarq`.

# Installation for development
If you'd like to contribute to SuperstaQ, below are the instructions for installation. Note, **if you are working on multiple clients** (e.g., `qiskit-superstaq` and `cirq-superstaq`), you do not need to clone the repository multiple times or set up multiple virtual environments, but you must install the client-specific requirements in each client directory.

<details>
<summary> <h3> <code>qiskit-superstaq</code> </h3> </summary>
  
  ```console
  git clone git@github.com:SupertechLabs/superstaq-client.git
  python3 -m venv venv_superstaq
  source venv_superstaq/bin/activate
  cd superstaq-client/qiskit-superstaq
  pip install -e ."[dev]"
  ```
</details>

<details>
<summary> <h3> <code>cirq-superstaq</code> </h3> </summary>
  
  ```console
  git clone git@github.com:SupertechLabs/superstaq-client.git
  python3 -m venv venv_superstaq
  source venv_superstaq/bin/activate
  cd superstaq-client/cirq-superstaq
  pip install -e ."[dev]"
  ```
</details>

<details>
<summary> <h3> <code>general-superstaq</code> </h3> </summary>
  
  ```console
  git clone git@github.com:SupertechLabs/superstaq-client.git
  python3 -m venv venv_superstaq
  source venv_superstaq/bin/activate
  cd superstaq-client/general-superstaq
  pip install -e ."[dev]"
  ```
</details>

<details>
<summary> <h3> <code>supermarq</code> </h3> </summary>
  
  ```console
  git clone git@github.com:SupertechLabs/superstaq-client.git
  python3 -m venv venv_superstaq
  source venv_superstaq/bin/activate
  cd superstaq-client/supermarq-benchmarks
  pip install -e ."[dev]"
  ```
</details>

# Documentation 
For more information on getting started, check out [our documentation site](https://docs-superstaq.readthedocs.io/)!

# License
SuperstaQ is licensed under the Apache License 2.0. See our [LICENSE](https://github.com/SupertechLabs/superstaq-client/blob/main/LICENSE) file for more details.

# Contact Us
If you'd like to reach out to a member of our team, please email us at info@super.tech.
