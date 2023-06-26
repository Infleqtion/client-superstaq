#!/usr/bin/env python3
# pylint: disable=missing-function-docstring
import sys

import general_superstaq.check

if __name__ == "__main__":
    exit(general_superstaq.check.flake8_.run(*sys.argv[1:]))
