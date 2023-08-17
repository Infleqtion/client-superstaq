#!/usr/bin/env python3

import sys
import textwrap
from typing import List, Optional

from general_superstaq.check import (
    build_docs,
    check_utils,
    configs,
    coverage_,
    flake8_,
    format_,
    mypy_,
    pylint_,
    requirements,
)


CHECK_TYPE = Literal[
    "configs",
    "format",
    "flake8",
    "pylint",
    "mypy",
    "coverage",
    "requirements",
    "build_docs",
]


def run(
    *args: str,
    sphinx_paths: Optional[List[str]] = None,
    skip: Optional[List[CHECK_TYPE]] = None,
) -> int:
    """Runs all checks on the repository.

    Args:
        *args: Command line arguments.
        sphinx_paths: List of sphinx paths strings (used for building docs).
        skip: List of checks to skip.

    Returns:
        Terminal exit code. 0 indicates success, while any other integer indicates a test failure.
    """

    parser = check_utils.get_file_parser()
    parser.description = textwrap.dedent(
        """
        Runs all checks on the repository.
        Exits immediately upon any failure unless passed one of -f, -F, or --force as an argument.
        This script exits with a succeeding exit code if and only if all checks pass.
        """
    )

    parser.add_argument(
        "-f",
        action="store_true",
        dest="force_formats",
        help="'Soft force' ~ continue past (i.e. do not exit after) failing format checks.",
    )
    parser.add_argument(
        "-F",
        "--force",
        action="store_true",
        dest="force_all",
        help="'Hard force' ~ continue past (i.e. do not exit after) all failing checks.",
    )

    parsed_args, _ = parser.parse_known_intermixed_args(args)
    if parsed_args.revisions is not None:
        # print info about incremental files once now, rather than in each check
        _ = check_utils.extract_files(parsed_args, silent=False)

    default_mode = not parsed_args.files and parsed_args.revisions is None
    checks_failed = 0

    args_to_pass = [arg for arg in args if arg not in ("-f", "-F", "--force")]

    skip = skip or []

    # run formatting checks
    # silence most checks to avoid printing duplicate info about incremental files
    # silencing does not affect warnings and errors
    exit_on_failure = not (parsed_args.force_formats or parsed_args.force_all)
    if "configs" not in skip:
        checks_failed |= configs.run(exit_on_failure=exit_on_failure, silent=True)
    if "format" not in skip:
        checks_failed |= format_.run(*args_to_pass, exit_on_failure=exit_on_failure, silent=True)
    if "flake8" not in skip:
        checks_failed |= flake8_.run(*args_to_pass, exit_on_failure=exit_on_failure, silent=True)
    if "pylint" not in skip:
        checks_failed |= pylint_.run(*args_to_pass, exit_on_failure=exit_on_failure, silent=True)

    # run typing and coverage checks
    exit_on_failure = not parsed_args.force_all
    if "mypy" not in skip:
        checks_failed |= mypy_.run(*args_to_pass, exit_on_failure=exit_on_failure, silent=True)
    if "coverage" not in skip:
        checks_failed |= coverage_.run(*args_to_pass, exit_on_failure=exit_on_failure, silent=True)

    # check that all pip requirements files are in order
    if "requirements" not in skip:
        checks_failed |= requirements.run(exit_on_failure=exit_on_failure)

    if default_mode and "build_docs" not in skip:
        # checks that the docs build
        checks_failed |= build_docs.run(exit_on_failure=exit_on_failure, sphinx_paths=sphinx_paths)

    return checks_failed


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
