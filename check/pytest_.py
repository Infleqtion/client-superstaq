#!/usr/bin/env python3

import sys
from typing import List, Optional

import general_superstaq.check

if __name__ == "__main__":

    if "--notebook" in sys.argv[1:]:
        EXCLUDE: Optional[List[str]] = [
            "examples/aqt.ipynb",
        ]
    else:
        EXCLUDE = None

    exit(general_superstaq.check.pytest_.run(*sys.argv[1:], exclude=EXCLUDE))
