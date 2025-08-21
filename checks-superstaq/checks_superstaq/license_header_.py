#!/usr/bin/env python3
from __future__ import annotations

import sys
import subprocess
import textwrap
from collections.abc import Iterable

from checks_superstaq import check_utils
from pathlib import Path
from typing import Any


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: str | Iterable[str] = "*.py",
    exclude: str | Iterable[str] = "*.ipynb",
    silent: bool = False,
) -> int:
    """Runs the 'licensepy format' tool on the repository (license header check/formatting).

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
        Runs 'licensepy format' on the repository (adds license headers to source code)
        """
    )
    parser.add_argument("--fix", action="store_true", help="Apply changes to files.")

    parsed_args, args_to_pass = parser.parse_known_intermixed_args(args)

    if "add_license_header" in parsed_args.skip:
        return 0

    if not parsed_args.fix:
        args_to_pass.append("--dry-run")

    files = check_utils.extract_files(parsed_args, include, exclude, silent)

    if files:
        return_code = subprocess.call(
            ["licensepy", "format", *files, *args_to_pass],
            cwd=check_utils.root_dir,
        )
        if return_code > 0:
            command = "./checks/license_header --fix"
            text = (
                f"Run '{command}' (from the repo root directory) to format "
                "files with correct license header."
            )
            print(check_utils.warning(text))  # noqa: T201
            return 1

        return return_code

    return 0

if __name__ == "__main__":
    sys.exit(run(*sys.argv[1:]))