#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Iterable

from checks_superstaq import check_utils

os.environ["FORCE_COLOR"] = "1"


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: str | Iterable[str] = "*.py",
    exclude: str | Iterable[str] = (),
    silent: bool = False,
) -> int:
    """Runs 'ruff check' on the repository (formatting check).

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
        Runs 'ruff check' on the repository (linting check).
        """
    )

    parsed_args, args_to_pass = parser.parse_known_intermixed_args(args)
    if "lint" in parsed_args.skip:
        return 0

    files = check_utils.extract_files(parsed_args, include, exclude, silent)

    if files:
        return subprocess.call(
            ["python", "-m", "ruff", "check", *files, *args_to_pass], cwd=check_utils.root_dir
        )

    return 0


if __name__ == "__main__":
    sys.exit(run(*sys.argv[1:]))
