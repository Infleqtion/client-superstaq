#!/usr/bin/env python3
from __future__ import annotations

import sys

import checks_superstaq as checks

if __name__ == "__main__":
    sys.exit(
        checks.build_docs.run(
            *sys.argv[1:],
        )
    )  
