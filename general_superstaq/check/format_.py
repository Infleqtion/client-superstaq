#!/usr/bin/env python3

import os
import subprocess
import sys
import textwrap
from typing import Iterable, Union

from general_superstaq.check import check_utils


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: Union[str, Iterable[str]] = ("*.py", "*.ipynb"),
    exclude: Union[str, Iterable[str]] = "",
    silent: bool = False,
) -> int:

    parser = check_utils.get_file_parser()
    parser.description = textwrap.dedent(
        """
        Runs black on the repository (formatting check).
        """
    )

    parser.add_argument("--apply", action="store_true", help="Apply changes to files.")

    parsed_args, args_to_pass = parser.parse_known_intermixed_args(args)
    files = check_utils.extract_files(parsed_args, include, exclude, silent)

    args_to_pass = ["--color", "--line-length=100"] + args_to_pass
    if not parsed_args.apply:
        args_to_pass = ["--diff", "--check"] + args_to_pass

    returncode = subprocess.call(["black", *files, *args_to_pass], cwd=check_utils.root_dir)

    if returncode == 1:
        # some files should be reformatted, but there don't seem to be any bona fide errors
        this_file = os.path.relpath(__file__)
        print(f"Run '{this_file} --apply' to format the files.")

    return returncode


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
