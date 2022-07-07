#!/usr/bin/env python3

import sys

import applications_superstaq.check

if __name__ == "__main__":
    exit(applications_superstaq.check.coverage_.run(*sys.argv[1:]))
