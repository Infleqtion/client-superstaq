<p align="center">
  <img src="./docs/source/_static/logos/Superstaq_color.png#gh-light-mode-only">
  <img src="./docs/source/_static/logos/Superstaq_white.png#gh-dark-mode-only">
</p>

# Welcome to Superstaq!
This repository is the home of the Superstaq development team's open-source work, which includes:
* Our quantum software platform that is optimized across the quantum stack and enables users to write quantum programs in Cirq or Qiskit and target a variety of quantum computers and simulators. Read more about it [here](https://www.infleqtion.com/superstaq).
* Our quantum benchmarking suite. More [here](https://github.com/Infleqtion/client-superstaq/tree/main/supermarq-benchmarks).

<p align="center"><img src="docs/source/_static/svg/code.svg"></p>

# Installation for users
For installation instructions for users of Superstaq, check out [our documentation site](https://superstaq.readthedocs.io/)! In short, you can install any of our packages by doing `pip install <package>` in a terminal, where `<package>` is `qiskit-superstaq`, `cirq-superstaq`, `general-superstaq`, or `supermarq`.

# Installation for development
If you'd like to contribute to Superstaq, below are the instructions for installation. Note, **if you are working on multiple clients** (e.g., `qiskit-superstaq` and `cirq-superstaq`), you do not need to clone the repository multiple times or set up multiple virtual environments, but you must install the client-specific requirements in each client directory.

<details>
<summary> <h3> <code>qiskit-superstaq</code> </h3> </summary>
  
  ```console
  git clone git@github.com:Infleqtion/client-superstaq.git
  python3 -m venv venv_superstaq
  source venv_superstaq/bin/activate
  cd client-superstaq/qiskit-superstaq
  python3 -m pip install -e ".[dev]"
  ```
</details>

<details>
<summary> <h3> <code>cirq-superstaq</code> </h3> </summary>
  
  ```console
  git clone git@github.com:Infleqtion/client-superstaq.git
  python3 -m venv venv_superstaq
  source venv_superstaq/bin/activate
  cd  client-superstaq/cirq-superstaq
  python3 -m pip install -e ."[dev]"
  ```
</details>

<details>
<summary> <h3> <code>general-superstaq</code> </h3> </summary>
  
  ```console
  git clone git@github.com:Infleqtion/client-superstaq.git
  python3 -m venv venv_superstaq
  source venv_superstaq/bin/activate
  cd  client-superstaq/general-superstaq
  python3 -m pip install -e ."[dev]"
  ```
</details>

<details>
<summary> <h3> <code>supermarq</code> </h3> </summary>
  
  ```console
  git clone git@github.com:Infleqtion/client-superstaq.git
  python3 -m venv venv_superstaq
  source venv_superstaq/bin/activate
  cd  client-superstaq/supermarq-benchmarks
  python3 -m pip install -e ."[dev]"
  ```
</details>

# Documentation 
For more information on getting started, check out [our documentation site](https://superstaq.readthedocs.io/)!

# License
Superstaq is licensed under the Apache License 2.0. See our [LICENSE](https://github.com/Infleqtion/client-superstaq/blob/main/LICENSE) file for more details.

# Contact Us
If you'd like to reach out to a member of our team, please email us at superstaq@infleqtion.com or join our [Slack workspace](https://join.slack.com/t/superstaq/shared_invite/zt-1wr6eok5j-fMwB7dPEWGG~5S474xGhxw).

<p align="center" style="padding: 50px">
  <img src="./docs/source/_static/logos/Infleqtion_logo.png#gh-light-mode-only" style="width: 20%">
  <img src="./docs/source/_static/logos/Infleqtion_logo_white.png#gh-dark-mode-only" style="width: 20%">
</p>
