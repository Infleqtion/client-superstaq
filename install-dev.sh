#!/usr/bin/env sh
pip install -e $(ls ./*/pyproject.toml | sed 's/\/pyproject.toml/\[dev\]/')
