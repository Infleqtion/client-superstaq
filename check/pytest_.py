#!/usr/bin/env python3

import sys

import applications_superstaq.check

if __name__ == "__main__":
    exit(applications_superstaq.check.pytest_.run(*sys.argv[1:]))
