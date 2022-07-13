#!/usr/bin/env python3

import subprocess
import sys
import textwrap
from typing import Iterable, Union

from applications_superstaq.check import check_utils


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: Union[str, Iterable[str]] = "*.py",
    exclude: Union[str, Iterable[str]] = "",
    silent: bool = False,
) -> int:

    parser = check_utils.get_file_parser()
    parser.description = textwrap.dedent(
        """
        Runs pylint on the repository (formatting check).
        NOTE: Only checks incrementally changed files by default.
        """
    )

    parser.add_argument("-a", "--all", action="store_true", help="Run pylint on the entire repo.")

    parsed_args, args_to_pass = parser.parse_known_intermixed_args(args)
    files = check_utils.extract_files(parsed_args, include, exclude, silent)

    if parsed_args.all:
        files += check_utils.get_tracked_files(include, exclude)

    return subprocess.call(["pylint", *files, *args_to_pass], cwd=check_utils.root_dir)


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
