#!/usr/bin/env python3

import sys

import general_superstaq.check

if __name__ == "__main__":
    args = sys.argv[1:]
    args += ["-x", "examples/aqt.ipynb"]
    exit(general_superstaq.check.pytest_.run(*args))
