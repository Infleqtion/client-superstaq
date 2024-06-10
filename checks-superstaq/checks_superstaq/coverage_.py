#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Iterable

from checks_superstaq import check_utils


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: str | Iterable[str] = "*.py",
    exclude: str | Iterable[str] = "*_integration_test.py",
    silent: bool = False,
) -> int:
    """Checks to make sure that all code is covered by unit tests.

    Args:
        *args: Command line arguments.
        include: Glob(s) indicating which tracked files to consider (e.g. "*.py").
        exclude: Glob(s) indicating which tracked files to skip (e.g. "*integration_test.py").
        silent: If True, restrict printing to warning and error messages.

    Returns:
        Terminal exit code. 0 indicates success, while any other integer indicates a test failure.
    """

    parser = check_utils.get_check_parser()
    parser.add_argument(
        "--modular",
        action="store_true",
        help="Check that each file is covered by its own test file.",
    )
    parser.description = textwrap.dedent(
        """
        Checks to make sure that all code is covered by unit tests.
        Fails if any pytest fails or if coverage is not 100%.
        Ignores integration tests and files in the [repo_root]/examples directory.
        Passes --disable-socket to coverage, unless running with --enable-socket.
        """
    )

    parsed_args, pytest_args = parser.parse_known_intermixed_args(args)
    if "coverage" in parsed_args.skip:
        return 0

    files = check_utils.extract_files(parsed_args, include, exclude, silent)
    silent = silent or not (parsed_args.files or parsed_args.revisions)

    if not parsed_args.modular:
        test_files = check_utils.get_test_files(files, exclude=exclude, silent=silent)
        if not test_files:
            print("No test files to check for pytest and coverage.")
            return 0
        return _run_on_files(files, test_files, pytest_args, exclude, silent)

    else:
        # run checks on individual files
        exit_codes = {}
        for file in files:
            test_files = check_utils.get_test_files([file], exclude=exclude, silent=silent)
            if not test_files:
                continue
            exit_codes[file] = _run_on_files([file], test_files, pytest_args, exclude, silent)

        if not exit_codes:
            print("No test files to check for pytest and coverage.")
            return 0

        # print warnings for files that are not covered
        for file, exit_code in exit_codes.items():
            if exit_code:
                check_utils.warning(f"Coverage failed for {file}.")

        return sum(exit_codes.values())


def _run_on_files(
    files: list[str],
    test_files: list[str],
    pytest_args: list[str],
    exclude: str | Iterable[str],
    silent: bool,
) -> int:
    """Run coverage tests on the specified files."""

    coverage_arg = "--include=" + ",".join(files)
    test_returncode = subprocess.call(
        [
            "python",
            "-m",
            "coverage",
            "run",
            coverage_arg,
            "-m",
            "pytest",
            *test_files,
            *pytest_args,
        ],
        cwd=check_utils.root_dir,
    )

    coverage_returncode = subprocess.call(
        ["python", "-m", "coverage", "report", "--precision=2"],
        cwd=check_utils.root_dir,
    )

    if test_returncode:
        print(check_utils.failure("TEST FAILURE!"))
        return test_returncode

    if coverage_returncode:
        print(check_utils.failure("COVERAGE FAILURE!"))
        return coverage_returncode

    print(check_utils.success("TEST AND COVERAGE SUCCESS!"))
    return 0


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
