#!/usr/bin/env python3
from __future__ import annotations

import sys

import checks_superstaq as checks

if __name__ == "__main__":
    args = [*sys.argv[1:], "--ruff"]
    if sys.version_info.minor >= 12:
        args += ["--sysmon"]
    sys.exit(checks.all_.run(*args))
