#!/usr/bin/env python3

import argparse
import sys
import textwrap
from typing import Iterable, Optional

from applications_superstaq.check import (
    build_docs,
    check_utils,
    coverage_,
    flake8_,
    format_,
    mypy_,
    pylint_,
    requirements,
)


@check_utils.extract_file_args
def run(
    *args: str,
    files: Optional[Iterable[str]] = None,
    parser: argparse.ArgumentParser = check_utils.get_file_parser()
) -> int:

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
        "-i",
        "--incremental",
        dest="revisions",
        nargs="*",
        action="extend",
        help="Run checks in incremental mode.",
    )
    parsed_args = parser.parse_intermixed_args(args)
    revisions = parsed_args.revisions

    default_mode = not files and revisions is None
    checks_failed = 0

    # run formatting checks
    exit_on_failure = not (parsed_args.force_formats or parsed_args.force_all)
    checks_failed |= format_.run(files=files, revisions=revisions, exit_on_failure=exit_on_failure)
    checks_failed |= flake8_.run(files=files, revisions=revisions, exit_on_failure=exit_on_failure)
    # pylint is slow, so always run pylint in incremental mode
    checks_failed |= pylint_.run(
        files=files,
        revisions=[] if default_mode else revisions,
        exit_on_failure=exit_on_failure,
    )

    # run typing and coverage checks
    exit_on_failure = not parsed_args.force_all
    # typing changes are likely to have side effects, so always run mypy on the entire repo
    checks_failed |= mypy_.run(revisions=revisions, exit_on_failure=exit_on_failure)
    checks_failed |= coverage_.run(
        files=files,
        revisions=revisions,
        exit_on_failure=exit_on_failure,
        suppress_warnings=default_mode,
    )

    # check that all pip requirements files are in order
    checks_failed |= requirements.run(exit_on_failure=exit_on_failure)

    if default_mode:
        # checks that the docs build
        checks_failed |= build_docs.run(exit_on_failure=exit_on_failure)

    return checks_failed


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
