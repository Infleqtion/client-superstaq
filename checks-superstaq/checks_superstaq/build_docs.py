#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from checks_superstaq import check_utils


@check_utils.enable_exit_on_failure
def run(*args: str, sphinx_paths: list[str] | None = None) -> int:
    """Checks that the docs build successfully.

    Args:
        *args: Command line arguments.
        sphinx_paths: List of sphinx paths (passed to `sphinx-apidoc`).

    Returns:
        Terminal exit code. 0 indicates success, while any other integer indicates a test failure.
    """

    parser = check_utils.get_check_parser(no_files=True)
    parser.description = textwrap.dedent(
        """
        Checks that the docs build successfully.
        """
    )
    parsed_args, _ = parser.parse_known_intermixed_args(args)
    if "build_docs" in parsed_args.skip:
        return 0

    docs_dir = os.path.join(check_utils.root_dir, "docs")

    if sphinx_paths:
        returncode = 0
        for path in sphinx_paths:
            returncode |= subprocess.call(
                ["sphinx-apidoc", "-f", "-o", "source", path, f"{path}/*_test.py"], cwd=docs_dir
            )
        return subprocess.call(["sphinx-build", "source", "build/html"], cwd=docs_dir)
    else:
        print(check_utils.warning("No docs to build."))
        return 0


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
