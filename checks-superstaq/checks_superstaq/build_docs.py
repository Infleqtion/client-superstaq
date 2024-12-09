#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from checks_superstaq import check_utils


@check_utils.enable_exit_on_failure
def run(*args: str) -> int:
    """Checks that the docs build successfully.

    Args:
        *args: Command line arguments.

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
    if not os.path.isdir(os.path.join(docs_dir, "source")):
        print(check_utils.warning("No docs to build."))
        return 0
    return subprocess.call(
        ["sphinx-build", "source", "build/html", "--fail-on-warning", "--keep-going"], cwd=docs_dir
    )


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
