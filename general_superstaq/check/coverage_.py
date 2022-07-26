#!/usr/bin/env python3

import subprocess
import sys
import textwrap
from typing import Iterable, Union

from general_superstaq.check import check_utils


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: Union[str, Iterable[str]] = "*.py",
    exclude: Union[str, Iterable[str]] = "*_integration_test.py",
    silent: bool = False,
) -> int:

    parser = check_utils.get_file_parser()
    parser.description = textwrap.dedent(
        """
        Checks to make sure that all code is covered by unit tests.
        Fails if any pytest fails or if coverage is not 100%.
        Ignores integration tests and files in the [repo_root]/examples directory.
        Passes --disable-socket to coverage, unless running with --enable-socket.
        """
    )

    parser.add_argument("--enable-socket", action="store_true", help="Force-enable socket.")

    parsed_args, args_to_pass = parser.parse_known_intermixed_args(args)
    files = check_utils.extract_files(parsed_args, include, exclude, silent)

    silent = silent or not (parsed_args.files or parsed_args.revisions)
    test_files = check_utils.get_test_files(*files, exclude=exclude, silent=silent)

    if not test_files:
        print("No test files to check for pytest and coverage.")
        return 0

    pytest_args = ["--disable-socket"] if not parsed_args.enable_socket else []

    args_to_pass.append("--include=" + ",".join(files))
    test_returncode = subprocess.call(
        ["coverage", "run", *args_to_pass, "-m", "pytest", *test_files, *pytest_args],
        cwd=check_utils.root_dir,
    )

    coverage_returncode = subprocess.call(
        ["coverage", "report", "--precision=2"], cwd=check_utils.root_dir
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
