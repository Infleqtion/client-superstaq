#!/usr/bin/env python3
import subprocess
import sys
import textwrap
from typing import Iterable, Union

from checks_superstaq import check_utils


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: Union[str, Iterable[str]] = "*.py",
    exclude: Union[str, Iterable[str]] = "*_integration_test.py",
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
    parser.description = textwrap.dedent(
        """
        Checks to make sure that all code is covered by unit tests.
        Fails if any pytest fails or if coverage is not 100%.
        Ignores integration tests and files in the [repo_root]/examples directory.
        Passes --disable-socket to coverage, unless running with --enable-socket.
        """
    )

    parser.add_argument("--enable-socket", action="store_true", help="Force-enable socket.")

    parsed_args, pytest_args = parser.parse_known_intermixed_args(args)
    if "coverage" in parsed_args.skip:
        return 0

    files = check_utils.extract_files(parsed_args, include, exclude, silent)

    silent = silent or not (parsed_args.files or parsed_args.revisions)
    test_files = check_utils.get_test_files(files, exclude=exclude, silent=silent)

    if not test_files:
        print("No test files to check for pytest and coverage.")
        return 0

    if not parsed_args.enable_socket:
        pytest_args.append("--disable-socket")

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
