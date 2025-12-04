#!/usr/bin/env python3
# Copyright 2025 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
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
    parser.add_argument(
        "--branch",
        action="store_true",
        help="Also require all branches to be covered (same as `coverage run --branch`).",
    )
    parser.add_argument(
        "--sysmon",
        action="store_true",
        help="Enable the `COVERAGE_CORE=sysmon` env variable for faster coverage (requires "
        "Python 3.12 or higher). Note: using the `--branch` option alongside `--sysmon` may require"
        " additional configuration for efficient execution.",
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

    # Enable threading setting before other args so -n can be overwritten
    default_threads = (
        "auto"
        if not parsed_args.files
        and not parsed_args.modular
        and parsed_args.revisions is None
        and "-s" not in pytest_args
        else "0"
    )
    pytest_args = [f"-n={default_threads}", *pytest_args]

    coverage_args = []
    if parsed_args.sysmon and sys.version_info.minor >= 12:
        os.environ["COVERAGE_CORE"] = "sysmon"

    if parsed_args.branch:
        coverage_args.append("--branch")

    files = check_utils.extract_files(parsed_args, include, exclude, silent)
    silent = silent or not (parsed_args.files or parsed_args.revisions)

    test_files = check_utils.get_test_files(files, exclude=exclude, silent=silent)
    if not test_files:
        print("No test files to check for pytest and coverage.")  # noqa: T201
        return 0

    if not parsed_args.modular:
        test_returncode = _run_on_files(files, test_files, coverage_args, pytest_args)
        return _report(test_returncode)

    # Run checks on individual files, skipping repeats
    subprocess.check_call(["python", "-m", "coverage", "erase"])

    # Move test files to the end of the file list, so if both "x.py" and "x_test.py" are in `files`
    # both will be included in the coverage report
    files.sort(
        key=lambda file: file.endswith("_test.py") or os.path.basename(file).startswith("test_")
    )
    coverage_args.append("--append")
    test_returncode = 0

    while files:
        file = files.pop(0)
        test_files = check_utils.get_test_files([file], exclude=exclude, silent=True)

        if not test_files:
            # Arguably we should fail the modular coverage check if no test file exists, but for now
            # just skip
            continue

        # File(s) to include in coverage report. If both "x.py" and "x_test.py" are passed, require
        # full coverage on each; otherwise only require coverage on whichever was passed (even
        # though both are used to run the test)
        files_requiring_coverage = [file]
        for test_file in test_files:
            if test_file in files:
                files_requiring_coverage.append(test_file)
                files.remove(test_file)

        test_returncode |= _run_on_files(
            files_requiring_coverage, test_files, coverage_args, pytest_args
        )

    return _report(test_returncode)


def _run_on_files(
    files_requiring_coverage: list[str],
    test_files: list[str],
    coverage_args: list[str],
    pytest_args: list[str],
) -> int:
    """Helper function to run coverage tests on the given files with the given pytest arguments."""
    coverage_args = ["--include=" + ",".join(files_requiring_coverage), *coverage_args]

    test_returncode = subprocess.call(
        [
            "python",
            "-m",
            "coverage",
            "run",
            *coverage_args,
            "-m",
            "pytest",
            *test_files,
            *pytest_args,
        ],
        cwd=check_utils.root_dir,
    )
    return test_returncode


def _report(test_returncode: int) -> int:
    subprocess.call(
        ["python", "-m", "coverage", "combine"],
        cwd=check_utils.root_dir,
        stdout=subprocess.DEVNULL,
    )

    coverage_returncode = subprocess.call(
        ["python", "-m", "coverage", "report", "--precision=2"],
        cwd=check_utils.root_dir,
    )

    if test_returncode:
        print(check_utils.failure("TEST FAILURE!"))  # noqa: T201
        return test_returncode

    if coverage_returncode:
        print(check_utils.failure("COVERAGE FAILURE!"))  # noqa: T201
        return coverage_returncode

    print(check_utils.success("TEST AND COVERAGE SUCCESS!"))  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(run(*sys.argv[1:]))
