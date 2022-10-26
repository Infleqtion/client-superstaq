#!/usr/bin/env python3

import subprocess
import sys
import textwrap
from typing import Callable, Iterable, Optional, Tuple, Union

from general_superstaq.check import check_utils


@check_utils.enable_exit_on_failure
def run(
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
        help="Run pytest on *_integration_test.py files, ignoring dev_tools/*.",
    )

    parser.add_argument("--enable-socket", action="store_true", help="Force-enable socket.")

    parsed_args, args_to_pass = parser.parse_known_intermixed_args(args)
    include, exclude = _get_file_search_options(
        parsed_args.notebook, parsed_args.integration, include, exclude
    )
    files = check_utils.extract_files(parsed_args, include, exclude, silent)

    if parsed_args.notebook:
        args_to_pass += ["--nbmake"]

    if not parsed_args.integration and not parsed_args.enable_socket:
        args_to_pass += ["--disable-socket"]

    if parsed_args.integration and integration_setup:
        integration_setup()

    return subprocess.call(["pytest", *files, *args_to_pass], cwd=check_utils.root_dir)


def _get_file_search_options(
    notebook_mode: bool,
    integration_mode: bool,
    include: Optional[Union[str, Iterable[str]]],
    exclude: Optional[Union[str, Iterable[str]]],
) -> Tuple[Union[str, Iterable[str]], Union[str, Iterable[str]]]:
    """If either of the include/exclude options are None, set them to reasonable defaults."""

    if notebook_mode:
        default_include = "*.ipynb"
        default_exclude = ""

    elif integration_mode:
        default_include = "*_integration_test.py"
        default_exclude = ""

    else:
        default_include = "*_test.py"
        default_exclude = "*_integration_test.py"

    if include is None:
        include = default_include
    if exclude is None:
        exclude = default_exclude

    return include, exclude


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
