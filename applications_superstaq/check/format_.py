#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import textwrap
from typing import Iterable, Optional

from applications_superstaq.check import check_utils

default_files_to_check = ("*.py", "*.ipynb")


@check_utils.enable_exit_on_failure
@check_utils.extract_file_args
@check_utils.enable_incremental(*default_files_to_check)
def run(
    *args: str,
    files: Optional[Iterable[str]] = None,
    parser: argparse.ArgumentParser = check_utils.get_file_parser(),
) -> int:

    parser.description = textwrap.dedent(
        """
        Runs black on the repository (formatting check).
        """
    )

    parser.add_argument("--apply", action="store_true", help="Apply changes to files.")
    parsed_args, unknown_args = parser.parse_known_intermixed_args(args)

    args = ("--color", "--line-length=100") + tuple(unknown_args)
    if not parsed_args.apply:
        args = ("--diff", "--check") + args

    if files is None:
        files = check_utils.get_tracked_files(*default_files_to_check)

    returncode = subprocess.call(["black", *args, *files], cwd=check_utils.root_dir)

    if returncode == 1:
        # some files should be reformatted, but there don't seem to be any bona fide errors
        this_file = os.path.relpath(__file__)
        print(f"Run '{this_file} --apply' to format the files.")

    return returncode


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
