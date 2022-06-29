#!/usr/bin/env python3

import argparse
import subprocess
import sys
import textwrap
from typing import Callable, Iterable, Optional, Union

from applications_superstaq.check import check_utils

default_files_to_check = ("*_test.py",)
default_exclude = ("*_integration_test.py",)


@check_utils.enable_exit_on_failure
@check_utils.extract_file_args
@check_utils.enable_incremental(*default_files_to_check, exclude=default_exclude)
def run(
    *args: str,
    files: Optional[Iterable[str]] = None,
    parser: argparse.ArgumentParser = check_utils.get_file_parser(),
    exclude: Optional[Union[str, Iterable[str]]] = default_exclude,
    integration_exclude: Optional[Union[str, Iterable[str]]] = "dev_tools/*",
    integration_setup: Optional[Callable] = None,
) -> int:

    parser.description = textwrap.dedent(
        """
        Runs pytest on the repository.
        Ignores integration tests unless running in integration mode.
        """
    )

    # notebook and integration tests are mutually exclusive
    exclusive_group = parser.add_mutually_exclusive_group()
    exclusive_group.add_argument(
        "--notebook",
        action="store_true",
        help="Run pytest on all *.ipynb files in the repository.",
    )
    exclusive_group.add_argument(
        "--integration",
        action="store_true",
        help="Run pytest on all *integration_test.py files in the repository.",
    )

    parser.add_argument(
        "--enable-socket",
        action="store_true",
        help="Force-enable socket (i.e. do not pass --disable-socket to pytest). "
        + "Enabled automatically if running in integration mode.",
    )

    parsed_args, unknown_args = parser.parse_known_intermixed_args(args)
    args = tuple(unknown_args)

    if parsed_args.notebook:
        args += ("--nbmake",)
        files = check_utils.get_tracked_files("**/*.ipynb", exclude=exclude)

    if parsed_args.integration:
        if integration_setup:
            integration_setup()

        files_to_add = check_utils.get_tracked_files(
            "*_integration_test.py",
            exclude=integration_exclude,
        )
        files = list(files) + files_to_add if files else files_to_add

    elif not parsed_args.enable_socket:
        args += ("--disable-socket",)

    if files is None:
        files = check_utils.get_tracked_files(*default_files_to_check, exclude=exclude)

    return subprocess.call(["pytest", *args, *files], cwd=check_utils.root_dir)


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
