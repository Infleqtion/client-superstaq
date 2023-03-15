#!/bin/bash

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
toplevel=$(git rev-parse --show-toplevel --path-format=absolute)

# This publishes general-superstaq to PyPI.
cd "$toplevel/general-superstaq"
python setup.py bdist_wheel
twine upload dist/* -u __token__ -p "$PYPI_API_KEY"

# This publishes qiskit-superstaq to PyPI.
cd "$toplevel/qiskit-superstaq"
python setup.py bdist_wheel
twine upload dist/* -u __token__ -p "$PYPI_API_KEY"

# This publishes cirq-superstaq to PyPI.
cd "$toplevel/cirq-superstaq"
python setup.py bdist_wheel
twine upload dist/* -u __token__ -p "$PYPI_API_KEY"

# This publishes supermarq to PyPI.
cd "$toplevel/supermarq-benchmarks"
python setup.py bdist_wheel
twine upload dist/* -u __token__ -p "$PYPI_API_KEY"
