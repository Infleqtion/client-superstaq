#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import textwrap
from typing import List, Optional

from general_superstaq.check import check_utils


@check_utils.enable_exit_on_failure
def run(*args: str, sphinx_paths: Optional[List[str]] = None) -> int:

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.description = textwrap.dedent(
        """
        Checks that the docs build successfully.
        """
    )
    parser.parse_args(args)  # placeholder parsing to enable printing help text

    docs_dir = os.path.join(check_utils.root_dir, "docs")
    make_file = os.path.join(docs_dir, "Makefile")
    if os.path.isfile(make_file):
        if sphinx_paths:
            for path in sphinx_paths:
                subprocess.run(
                    f"sphinx-apidoc -f -o source {path} {path}/*_test.py", shell=True, cwd=docs_dir
                )
        return subprocess.call(["make", *args, "html"], cwd=docs_dir)
    else:
        print(check_utils.warning("No docs to build."))
        return 0


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
