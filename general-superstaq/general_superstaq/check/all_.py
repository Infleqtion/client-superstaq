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

CHECK_LIST = [
    "configs",
    "format",
    "flake8",
    "pylint",
    "mypy",
    "coverage",
    "requirements",
    "build_docs",
]


def run(*args: str, sphinx_paths: Optional[List[str]] = None) -> int:
    """Runs all checks on the repository.

    Args:
        *args: Command line arguments.
        sphinx_paths: List of sphinx paths strings (used for building docs).

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
    parser.add_argument(
        "--skip",
        choices=CHECK_LIST,
        nargs="+",
        action="extend",
        default=[],
        help="The checks to skip.",
    )

    parsed_args, _ = parser.parse_known_intermixed_args(args)
    if parsed_args.revisions is not None:
        # print info about incremental files once now, rather than in each check
        _ = check_utils.extract_files(parsed_args, silent=False)

    default_mode = not parsed_args.files and parsed_args.revisions is None
    checks_failed = 0

    # run formatting checks
    # silence most checks to avoid printing duplicate info about incremental files
    # silencing does not affect warnings and errors
    exit_on_failure = not (parsed_args.force_formats or parsed_args.force_all)
    common_kwargs = dict(namespace=parsed_args, exit_on_failure=exit_on_failure, silent=True)
    if "configs" not in parsed_args.skip:
        checks_failed |= configs.run(**common_kwargs)
    if "format" not in parsed_args.skip:
        checks_failed |= format_.run(**common_kwargs)
    if "flake8" not in parsed_args.skip:
        checks_failed |= flake8_.run(**common_kwargs)
    if "pylint" not in parsed_args.skip:
        checks_failed |= pylint_.run(**common_kwargs)

    # run typing and coverage checks
    exit_on_failure = not parsed_args.force_all
    if "mypy" not in parsed_args.skip:
        checks_failed |= mypy_.run(**common_kwargs)
    if "coverage" not in parsed_args.skip:
        checks_failed |= coverage_.run(**common_kwargs)

    # check that all pip requirements files are in order
    if "requirements" not in parsed_args.skip:
        checks_failed |= requirements.run(exit_on_failure=exit_on_failure)

    if default_mode and "build_docs" not in parsed_args.skip:
        # checks that the docs build
        checks_failed |= build_docs.run(exit_on_failure=exit_on_failure, sphinx_paths=sphinx_paths)

    return checks_failed


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
