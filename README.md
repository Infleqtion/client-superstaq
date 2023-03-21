<img src="./docs/source/_static/logos/SuperstaQ_SSa-R00a_Mil.png">

# Welcome to SuperstaQ!

# Installation for development

For installation instructions for users of SuperstaQ, check out [our documentation site](https://docs-superstaq.readthedocs.io/).

If you'd like to contribute to SuperstaQ, below are the instructions for installation. Note, you do not need to clone the repository multiple times if you are working on different clients (e.g., `qiskit-superstaq` and `cirq-superstaq`), but you must install the client-specific requirements in each client directory.

If you have trouble running `pip install .[dev]`, try `pip install ."[dev]"`

<details>
<summary> <h3> <code>qiskit-superstaq</code> </h3> </summary>
  
  ```console
  git clone git@github.com:SupertechLabs/superstaq-client.git
  python3 -m venv venv_qiskit_superstaq
  source venv_qiskit_superstaq/bin/activate
  cd superstaq-client/qiskit-superstaq
  pip install qiskit-superstaq
  pip install .[dev]
  ```
</details>

<details>
<summary> <h3> <code>cirq-superstaq</code> </h3> </summary>
  
  ```console
  git clone git@github.com:SupertechLabs/superstaq-client.git
  python3 -m venv venv_cirq_superstaq
  source venv_cirq_superstaq/bin/activate
  cd superstaq-client/cirq-superstaq
  pip install qiskit-superstaq
  pip install .[dev]
  ```
</details>

<details>
<summary> <h3> <code>general-superstaq</code> </h3> </summary>
  
  ```console
  git clone git@github.com:SupertechLabs/superstaq-client.git
  python3 -m venv venv_general_superstaq
  source venv_general_superstaq/bin/activate
  cd superstaq-client/general-superstaq
  pip install general-superstaq
  pip install .[dev]
  ```
</details>

<details>
<summary> <h3> <code>supermarq</code> </h3> </summary>
  
  ```console
  git clone git@github.com:SupertechLabs/superstaq-client.git
  python3 -m venv venv_supermarq
  source venv_supermarq/bin/activate
  cd superstaq-client/supermarq-benchmarks
  pip install supermarq
  pip install .[dev]
  ```
</details>

# Documentation 
For more information on getting started, check out [our documentation site](https://docs-superstaq.readthedocs.io/)!

# License
SuperstaQ is licensed under the Apache License 2.0. See our [LICENSE](https://github.com/SupertechLabs/superstaq-client/blob/main/LICENSE) file for more details.

# Contact Us
If you'd like to reach out to a member of our team, please email us at info@super.tech.
