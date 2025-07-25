#!/usr/bin/env python3
from __future__ import annotations

import sys

import checks_superstaq as checks

if __name__ == "__main__":
    sys.exit(checks.lint_.run(*sys.argv[1:], include=("*.py", "*.ipynb")))
