<p align="center">
  <img src="./docs/source/_static/logos/Superstaq_color.png#gh-light-mode-only">
  <img src="./docs/source/_static/logos/Superstaq_white.png#gh-dark-mode-only">
</p>

<div align="center">

<a href="https://github.com/Infleqtion/client-superstaq/blob/main/LICENSE">
  <img src="https://img.shields.io/github/license/Infleqtion/client-superstaq?style=flat&logo=pypi&logoColor=white&labelColor=00b198&color=141a5e" alt="License" #gh-light-mode-only>
  <img src="https://img.shields.io/github/license/Infleqtion/client-superstaq?style=flat&logo=python&logoColor=white&labelColor=00b198&color=white" alt="License" #gh-dark-mode-only>
</a>

<a href="https://github.com/Infleqtion/client-superstaq">
  <img src="https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12%20|%203.13%20-141a5e?display_name=tag&style=flat&logo=pypi&logoColor=white&labelColor=00b198&color=141a5e" alt="Python Versions" #gh-light-mode-only>
  <img src="https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12%20|%203.13%20-white?style=flat&logo=python&logoColor=white&labelColor=00b198&color=white" alt="Python Versions" #gh-dark-mode-only>
</a>

<a href="https://github.com/Infleqtion/client-superstaq/releases">
  <img src="https://img.shields.io/github/v/release/Infleqtion/client-superstaq?display_name=tag&style=flat&logo=pypi&logoColor=white&labelColor=00b198&color=141a5e" alt="GitHub Release" #gh-light-mode-only>
  <img src="https://img.shields.io/github/v/release/Infleqtion/client-superstaq?display_name=tag&style=flat&logo=pypi&logoColor=white&labelColor=00b198&color=white" alt="GitHub Release" #gh-dark-mode-only>
</a>

<a href="https://github.com/Infleqtion/client-superstaq/actions/workflows/ci.yml">
  <img src="https://img.shields.io/github/actions/workflow/status/Infleqtion/client-superstaq/ci.yml?branch=main&style=flat&logo=github&logoColor=white&labelColor=00b198&color=141a5e" alt="CI Status" #gh-light-mode-only>
  <img src="https://img.shields.io/github/actions/workflow/status/Infleqtion/client-superstaq/ci.yml?branch=main&style=flat&logo=github&logoColor=white&labelColor=00b198&color=white" alt="CI Status" #gh-dark-mode-only>
</a>

<a href="https://superstaq.readthedocs.io/">
  <img src="https://img.shields.io/badge/Read%20the%20docs-a?style=flat&logo=read-the-docs&logoColor=white&labelColor=00b198&color=141a5e" alt="Read the Docs" #gh-light-mode-only>
  <img src="https://img.shields.io/badge/Read%20the%20docs-a?style=flat&logo=read-the-docs&logoColor=white&labelColor=00b198&color=white" alt="Read the Docs" #gh-dark-mode-only>
</a>

<a href="https://join.slack.com/t/superstaq/shared_invite/zt-1wr6eok5j-fMwB7dPEWGG~5S474xGhxw">
  <img src="https://img.shields.io/badge/Slack-slack?style=flat&logo=slack&logoColor=white&labelColor=00b198&color=141a5e" alt="Slack" #gh-light-mode-only>
  <img src="https://img.shields.io/badge/Slack-slack?style=flat&logo=slack&logoColor=white&labelColor=00b198&color=white" alt="Slack" #gh-dark-mode-only>
</a>
</div>

# Welcome to Superstaq

This repository is the home of the Superstaq development team's open-source work, which includes:

- Our quantum software platform that is optimized across the quantum stack and enables users to write quantum programs in Cirq or Qiskit and target a variety of quantum computers and simulators. Read more about it [here](https://www.infleqtion.com/superstaq).

<p align="center"><img src="docs/source/_static/svg/code.svg"></p>

# Installation for users

For installation instructions for users of Superstaq, check out [our documentation site](https://superstaq.readthedocs.io/)! In short, you can install any of our packages by doing `pip install <package>` in a terminal, where `<package>` is `qiskit-superstaq`, `cirq-superstaq`, or `general-superstaq`.

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
cd client-superstaq/cirq-superstaq
python3 -m pip install -e ".[dev]"
```

</details>

<details>
<summary> <h3> <code>general-superstaq</code> </h3> </summary>

```console
git clone git@github.com:Infleqtion/client-superstaq.git
python3 -m venv venv_superstaq
source venv_superstaq/bin/activate
cd client-superstaq/general-superstaq
python3 -m pip install -e ".[dev]"
```

</details>

# Documentation

For more information on getting started, check out [our documentation site](https://superstaq.readthedocs.io/)!

# License

Superstaq is licensed under the Apache License 2.0. See our [LICENSE](https://github.com/Infleqtion/client-superstaq/blob/main/LICENSE) file for more details.

# Contact Us

If you'd like to reach out to a member of our team, please email us at <superstaq@infleqtion.com> or join our [Slack workspace](https://join.slack.com/t/superstaq/shared_invite/zt-1wr6eok5j-fMwB7dPEWGG~5S474xGhxw).

<p align="center" style="padding: 50px">
  <img src="./docs/source/_static/logos/Infleqtion_logo.png#gh-light-mode-only" style="width: 20%">
  <img src="./docs/source/_static/logos/Infleqtion_logo_white.png#gh-dark-mode-only" style="width: 20%">
</p>
