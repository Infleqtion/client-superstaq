#!/usr/bin/env sh
cd $(dirname $0)
pip install -e $(ls ./*/pyproject.toml | sed 's/\/pyproject.toml/\[dev\]/')
