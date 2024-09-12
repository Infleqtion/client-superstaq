#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterable

from checks_superstaq import check_utils


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: str | Iterable[str] | None = None,
    exclude: str | Iterable[str] | None = None,
    integration_setup: Callable[[], None] | None = None,
    silent: bool = False,
) -> int:
    """Runs pytest on the repository.

    Args:
        *args: Command line arguments.
        include: Glob(s) indicating which tracked files to consider (e.g. "*.py").
        exclude: Glob(s) indicating which tracked files to skip (e.g. "*integration_test.py").
        integration_setup: Optional function to run before integration tests (for example,
            to set environmental variables).
        silent: If True, restrict printing to warning and error messages.

    Returns:
        Terminal exit code. 0 indicates success, while any other integer indicates a test failure.
    """

    parser = check_utils.get_check_parser()
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
    parser.add_argument(
        "--single_core",
        action="store_true",
        help="Run pytest without using xdist.",
    )

    parsed_args, args_to_pass = parser.parse_known_intermixed_args(args)
    if "pytest" in parsed_args.skip:
        return 0

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
        args_to_pass += ["--nbmake", "--force-enable-socket"]
    elif (parsed_args.integration) or (
        "--integration" not in args
        and any(re.match(r".*_integration_test\.py$", arg) for arg in args)
    ):
        args_to_pass += ["--force-enable-socket"]
    else:
        files = check_utils.get_test_files(files, exclude=exclude, silent=silent)

    if not files:
        return 0

    if parsed_args.notebook:
        # These tests spend most of their time waiting for the server, so allow more threads than we
        # have physical processors (within reason)
        nthreads = min(len(files), 16)
        nthreads = 0 if nthreads <= 1 else nthreads

        # Setting before other args so -n can be overwritten
        args_to_pass = [f"-n{nthreads}", *args_to_pass]

    elif (
        not parsed_args.files
        and not parsed_args.single_core
        and parsed_args.revisions is None
        and "-s" not in args_to_pass
    ):
        # enable threading
        # setting before other args so -n can be overwritten
        args_to_pass = ["-n=auto", *args_to_pass]

    if parsed_args.integration and integration_setup:
        integration_setup()

    return subprocess.call(
        ["python", "-m", "pytest", *files, *args_to_pass],
        cwd=check_utils.root_dir,
    )


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
