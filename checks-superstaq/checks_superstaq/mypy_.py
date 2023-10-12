#!/usr/bin/env python3
import subprocess
import sys
import textwrap
from typing import Iterable, Union

from checks_superstaq import check_utils


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: Union[str, Iterable[str]] = "*.py",
    exclude: Union[str, Iterable[str]] = (),
    silent: bool = False,
) -> int:
    """Runs mypy on the repository (typing check).

    Args:
        *args: Command line arguments.
        include: Glob(s) indicating which tracked files to consider (e.g. "*.py").
        exclude: Glob(s) indicating which tracked files to skip (e.g. "*integration_test.py").
        silent: If True, restrict printing to warning and error messages.

    Returns:
        Terminal exit code. 0 indicates success, while any other integer indicates a test failure.
    """

    parser = check_utils.get_check_parser()
    parser.description = textwrap.dedent(
        """
        Runs mypy on the repository (typing check).
        Ignores files in the [repo_root]/examples directory.
        """
    )

    parsed_args, args_to_pass = parser.parse_known_intermixed_args(args)
    if "mypy" in parsed_args.skip:
        return 0

    files = check_utils.extract_files(parsed_args, include, exclude, silent)

    return subprocess.call(
        ["python", "-m", "mypy", *files, *args_to_pass], cwd=check_utils.root_dir
    )


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
