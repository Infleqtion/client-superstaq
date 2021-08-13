This package is used to access SuperstaQ via a Web API through Qiskit. Qiskit programmers
can take advantage of the applications, pulse level optimization, and write-once-target-all


Please note that Python version `3.8` or higher is required. qiskit-superstaq and all of its
dependencies can be installed via:

```
git clone https://github.com/SupertechLabs/qiskit-superstaq.git
cd qiskit-superstaq
conda create -n qiskit-superstaq-env python=3.8
conda activate qiskit-superstaq-env
pip install -r dev-requirements.txt
pip install -e .
```

Make sure to always be in the qiskit-superstaq-env (via ``conda activate qiskit-superstaq-env``) when you're working on qiskit-superstaq.

Pandoc must be installed prior to building the docs. This can be done via ```apt install pandoc``` on Linux and ```brew install pandoc``` on Mac.

### Build the Docs

To build and display the docs locally:

```
./dev_tools/build_docs
open docs/build/html/index.html # For Mac
xdg-open docs/build/html/index.html # For Linux
```

### Unit Tests

In order to get unit tests to work locally, you must:
1. Have an IBM quantum account (with access to the ibm-q-startup provider)
2. Have an AWS Braket account, configured with these boto3 instructions https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration.
3. Clone and install qtrl
```
git clone git@github.com:SupertechLabs/qtrl.git
cd qtrl
pip install .
