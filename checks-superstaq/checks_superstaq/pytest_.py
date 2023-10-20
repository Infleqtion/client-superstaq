#!/usr/bin/env python3
import subprocess
import sys
import textwrap
from typing import Callable, Iterable, Optional, Union

from checks_superstaq import check_utils


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: Optional[Union[str, Iterable[str]]] = None,
    exclude: Optional[Union[str, Iterable[str]]] = None,
    integration_setup: Optional[Callable[[], None]] = None,
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

    parser.add_argument("--enable-socket", action="store_true", help="Force-enable socket.")

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
        args_to_pass += ["--nbmake"]
    elif not parsed_args.integration:
        files = check_utils.get_test_files(files, exclude=exclude, silent=silent)

    if not parsed_args.integration and not parsed_args.enable_socket:
        args_to_pass += ["--disable-socket"]

    if not files:
        return 0

    if parsed_args.integration and integration_setup:
        integration_setup()

    return subprocess.call(
        ["python", "-m", "pytest", *files, *args_to_pass], cwd=check_utils.root_dir
    )


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
