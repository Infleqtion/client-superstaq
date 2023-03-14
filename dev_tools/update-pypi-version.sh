#!/bin/bash

# This publishes cirq-superstaq to PyPI.
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
toplevel=$(git rev-parse --show-toplevel --path-format=absolute)

cd "$toplevel/cirq-superstaq"
python setup.py bdist_wheel
twine upload dist/* -u __token__ -p "$PYPI_API_KEY"
