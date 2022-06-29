#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import textwrap

from applications_superstaq.check import check_utils


@check_utils.enable_exit_on_failure
def run(
    *args: str, parser: argparse.ArgumentParser = check_utils.get_file_parser(add_files=False)
) -> int:

    parser.description = textwrap.dedent(
        """
        Checks that the docs build successfully.
        """
    )
    parser.parse_args(args)

    docs_dir = os.path.join(check_utils.root_dir, "docs")
    if os.path.isdir(docs_dir):
        return subprocess.call(["make", *args, "html"], cwd=docs_dir)
    else:
        print(check_utils.warning("No docs to build."))
        return 0


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
