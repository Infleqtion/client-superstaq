#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import textwrap

from applications_superstaq.check import check_utils


@check_utils.enable_exit_on_failure
def run(*args: str) -> int:

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.description = textwrap.dedent(
        """
        Checks that the docs build successfully.
        """
    )
    parser.parse_args(args)  # placeholder parsing to enable printing help text

    docs_dir = os.path.join(check_utils.root_dir, "docs")
    if os.path.isdir(docs_dir):
        return subprocess.call(["make", *args, "html"], cwd=docs_dir)
    else:
        print(check_utils.warning("No docs to build."))
        return 0


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
