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


def run(*args: str, sphinx_paths: Optional[List[str]] = None) -> int:

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

    parsed_args = parser.parse_intermixed_args(args)
    args_to_pass = parsed_args.files
    if parsed_args.revisions is not None:
        args_to_pass += ["-i", *parsed_args.revisions]

    default_mode = not parsed_args.files and parsed_args.revisions is None
    checks_failed = 0

    # run formatting checks
    # silence most checks to avoid printing duplicate info about incrmental files
    # silencing does not affect warnings and errors
    exit_on_failure = not (parsed_args.force_formats or parsed_args.force_all)
    checks_failed |= configs.run(exit_on_failure=exit_on_failure, silent=True)
    checks_failed |= format_.run(*args_to_pass, exit_on_failure=exit_on_failure, silent=False)
    checks_failed |= flake8_.run(*args_to_pass, exit_on_failure=exit_on_failure, silent=True)
    checks_failed |= pylint_.run(*args_to_pass, exit_on_failure=exit_on_failure, silent=True)

    # run typing and coverage checks
    exit_on_failure = not parsed_args.force_all
    checks_failed |= mypy_.run(*args_to_pass, exit_on_failure=exit_on_failure, silent=True)
    checks_failed |= coverage_.run(
        *args_to_pass, exit_on_failure=exit_on_failure, silent=default_mode
    )

    # check that all pip requirements files are in order
    checks_failed |= requirements.run(exit_on_failure=exit_on_failure)

    if default_mode:
        # checks that the docs build
        checks_failed |= build_docs.run(exit_on_failure=exit_on_failure, sphinx_paths=sphinx_paths)

    return checks_failed


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
