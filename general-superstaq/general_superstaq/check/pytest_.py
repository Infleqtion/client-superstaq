#!/usr/bin/env python3

import subprocess
import sys
import textwrap
from typing import Callable, Iterable, Optional, Union

from general_superstaq.check import check_utils


@check_utils.enable_exit_on_failure
def run(  # pylint: disable=missing-function-docstring
    *args: str,
    include: Optional[Union[str, Iterable[str]]] = None,
    exclude: Optional[Union[str, Iterable[str]]] = None,
    integration_setup: Optional[Callable[[], None]] = None,
    silent: bool = False,
) -> int:

    parser = check_utils.get_file_parser()
    parser.description = textwrap.dedent(
        """
        Runs pytest on the repository.
        By default, checks only *_test.py files, ignoring *_integration_test.py files.
        Passes --disable-socket to pytest, unless running with --integration or --enable-socket.
        """
    )

    # notebook and integration tests are mutually exclusive
    exclusive_group = parser.add_mutually_exclusive_group()
    exclusive_group.add_argument(
        "--notebook",
        action="store_true",
        help="Run pytest on *.ipynb files.",
    )
    exclusive_group.add_argument(
        "--integration",
        action="store_true",
        help="Run pytest on *_integration_test.py files.",
    )

    parser.add_argument("--enable-socket", action="store_true", help="Force-enable socket.")

    parsed_args, args_to_pass = parser.parse_known_intermixed_args(args)

    exclude = [exclude] if isinstance(exclude, str) else [] if exclude is None else list(exclude)
    if parsed_args.notebook:
        include = include or "*.ipynb"
    elif parsed_args.integration:
        include = include or "*_integration_test.py"
    else:
        include = include or "*.py"
        exclude.append("*_integration_test.py")

    files = check_utils.extract_files(parsed_args, include, exclude, silent)

    if parsed_args.notebook:
        args_to_pass += ["--nbmake"]
    elif not parsed_args.integration:
        files = check_utils.get_test_files(files, exclude=exclude, silent=silent)

    if not parsed_args.integration and not parsed_args.enable_socket:
        args_to_pass += ["--disable-socket"]

    if not files:
        return 0

    if parsed_args.integration and integration_setup:
        integration_setup()

    return subprocess.call(["pytest", *files, *args_to_pass], cwd=check_utils.root_dir)


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
