#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import sys
import textwrap
from typing import Iterable, List, Optional, Union

from applications_superstaq.check import check_utils

default_files_to_check = ("*.py",)
default_exclude = ("examples/*", "*_integration_test.py")


@check_utils.enable_exit_on_failure
@check_utils.extract_file_args
@check_utils.enable_incremental(*default_files_to_check, exclude=default_exclude)
def run(
    *args: str,
    files: Optional[Iterable[str]] = None,
    parser: argparse.ArgumentParser = check_utils.get_file_parser(),
    suppress_warnings: bool = False,
    exclude: Optional[Union[str, Iterable[str]]] = default_exclude,
) -> int:

    parser.description = textwrap.dedent(
        """
        Checks to make sure that all code is covered by unit tests.
        Fails if any pytest fails or if coverage is not 100%.
        Ignores integration tests and files in the [repo_root]/examples directory.
        """
    )
    parser.parse_args(args)

    if files is None:
        files = check_utils.get_tracked_files(*default_files_to_check, exclude=exclude)
        suppress_warnings = True

    test_files = _get_test_files(*files, exclude=exclude, suppress_warnings=suppress_warnings)

    if test_files:
        include_files = "--include=" + ",".join(files)
        test_returncode = subprocess.call(
            ["coverage", "run", *args, include_files, "-m", "pytest", *test_files],
            cwd=check_utils.root_dir,
        )

        coverage_returncode = subprocess.call(
            ["coverage", "report", "--precision=2"], cwd=check_utils.root_dir
        )

        if test_returncode:
            print(check_utils.failure("TEST FAILURE!"))
            exit(test_returncode)

        if coverage_returncode:
            print(check_utils.failure("COVERAGE FAILURE!"))
            exit(coverage_returncode)

        print(check_utils.success("TEST AND COVERAGE SUCCESS!"))

    else:
        print("No test files to check for pytest and coverage.")

    return 0


def _get_test_files(
    *files: str, exclude: Optional[Union[str, Iterable[str]]] = None, suppress_warnings: bool
) -> List[str]:
    """
    For the given files, identify all associated test files (i.e. files with the same name, but
    with a "_test.py" suffix).
    """
    should_include = check_utils.inclusion_filter(exclude)

    test_files = set()
    for file in files:
        if file.endswith("_test.py"):
            test_files.add(file)

        else:
            test_file = re.sub(r"\.py$", "_test.py", file)
            test_file_exists = os.path.isfile(os.path.join(check_utils.root_dir, test_file))
            if test_file_exists and should_include(test_file):
                test_files.add(test_file)
            elif not suppress_warnings:
                print(check_utils.warning(f"WARNING: no test file found for {file}"))

    return list(test_files)


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
