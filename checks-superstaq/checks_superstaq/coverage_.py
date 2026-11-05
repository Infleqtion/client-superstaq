#!/usr/bin/env python3
# Copyright 2026 Infleqtion, Inc.
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

import argparse
import concurrent.futures
import contextlib
import io
import os
import pathlib
import re
import subprocess
import sys
import textwrap
from collections.abc import Iterable, Iterator
from typing import cast

import coverage

from checks_superstaq import _coverage_protocol as protocol
from checks_superstaq import check_utils

# Pairs of files that require coverage and the test files that must cover them.
_TestPairs = list[tuple[list[str], list[str]]]


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
        Terminal exit code. 0 indicates success, while any other integer indicates a test or
        coverage failure.
    """
    parser = check_utils.get_check_parser()
    parser.add_argument(
        "--modular",
        action="store_true",
        help="Check that each file is covered by its own test file.",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="With --modular, run the selected test files in one pytest run instead of one process"
        " for each file and its test files, and approximate the coverage of each file by its test"
        " files with coverage contexts. Lines that run while pytest starts or a module is first"
        " imported are credited to every test file, and lines that only subprocesses of tests run"
        " are not credited. Only files below the coverage threshold are reported.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="The number of parallel jobs: without --modular, or with --modular --single-run, the"
        " number of pytest-xdist workers; with --modular, the number of files checked concurrently,"
        " each with its test files in its own process. 0 disables parallelism. Default with"
        " --modular: min((os.cpu_count() or 2) // 2, number of files to check), or of test files"
        " with --single-run.",
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
        " additional configuration for efficient execution. Ignored with --single-run, which uses"
        " coverage's C tracer.",
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
    if parsed_args.single_run and not parsed_args.modular:
        parser.error("--single-run requires --modular")

    if parsed_args.single_run:
        return _check_single_run(
            parser, parsed_args, pytest_args, include=include, exclude=exclude, silent=silent
        )

    # Enable threading setting before other args so -n can be overwritten
    default_threads = (
        parsed_args.jobs
        if not parsed_args.modular and parsed_args.jobs is not None
        else "auto"
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

    try:
        if parsed_args.modular:
            returncode = _run_modular(files, coverage_args, pytest_args, exclude, parsed_args.jobs)
        else:
            result = _run_on_files(files, test_files, coverage_args, pytest_args)
            returncode = result.returncode

    except BaseException as e:
        return _abort(e)

    return _report(returncode)


def _get_test_pairs(files: list[str], exclude: str | Iterable[str]) -> _TestPairs:
    """Pair the files that require coverage with the test files that must cover them."""
    # Move test files to the end of the file list, so if both "x.py" and "x_test.py" are in `files`
    # both will be included in the coverage report
    files = sorted(
        files,
        key=lambda file: file.endswith("_test.py") or os.path.basename(file).startswith("test_"),
    )

    # Build (files_requiring_coverage, test_files) pairs
    test_pairs: _TestPairs = []
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

        test_pairs.append((files_requiring_coverage, test_files))
    return test_pairs


def _run_modular(
    files: list[str],
    coverage_args: list[str],
    pytest_args: list[str],
    exclude: str | Iterable[str],
    num_jobs: int | None,
) -> int:
    """Run a coverage process for each file and its test files, in parallel unless num_jobs is 0."""
    subprocess.check_call([sys.executable, "-m", "coverage", "erase"], cwd=check_utils.root_dir)
    test_pairs = _get_test_pairs(files, exclude)

    test_returncode = 0
    if num_jobs == 0:
        # Run modular checks one at a time
        for files_requiring_coverage, test_files in test_pairs:
            result = _run_on_files(files_requiring_coverage, test_files, coverage_args, pytest_args)
            test_returncode |= result.returncode
        return test_returncode

    # Run modular checks concurrently
    max_workers = num_jobs or min((os.cpu_count() or 2) // 2, len(test_pairs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        jobs = [
            executor.submit(
                _run_on_files,
                files_requiring_coverage,
                test_files,
                coverage_args,
                pytest_args,
                capture_output=True,
            )
            for files_requiring_coverage, test_files in test_pairs
        ]
        for future in concurrent.futures.as_completed(jobs):
            result = future.result()
            print(result.stdout, end="")  # noqa: T201
            test_returncode |= result.returncode
    return test_returncode


def _check_single_run(
    parser: argparse.ArgumentParser,
    parsed_args: argparse.Namespace,
    pytest_args: list[str],
    *,
    include: str | Iterable[str],
    exclude: str | Iterable[str],
    silent: bool,
) -> int:
    """Run the modular check with ``--single-run``, as ``run`` does, and return its exit code."""
    files = check_utils.extract_files(parsed_args, include, exclude, silent)
    if any("::" in file for file in files):
        parser.error("--single-run checks whole test files, not test node IDs")
    silent = silent or not (parsed_args.files or parsed_args.revisions)

    # The paths of the files are relative to the root directory, where coverage also finds its
    # configuration.
    with _working_directory(check_utils.root_dir):
        if not check_utils.get_test_files(files, exclude=exclude, silent=silent):
            print("No test files to check for pytest and coverage.")  # noqa: T201
            return 0
        test_pairs = _get_test_pairs(files, exclude)
        try:
            data_file = _erase_coverage_data()
            test_returncode = _run_single_pytest(
                test_pairs, pytest_args, parsed_args.jobs, parsed_args.branch, data_file
            )
            coverage_returncode = _report_single_run(test_pairs, data_file)
        except BaseException as e:
            return _abort(e)
    return _finish(test_returncode, coverage_returncode)


def _erase_coverage_data() -> str:
    """Erase the configured coverage data, including parallel data files, and return its path."""
    measurement = coverage.Coverage()
    measurement.set_option("run:parallel", value=True)
    measurement.erase()
    return os.path.abspath(cast("str", measurement.get_option("run:data_file")))


def _run_single_pytest(
    test_pairs: _TestPairs,
    pytest_args: list[str],
    num_jobs: int | None,
    branch: bool,
    data_file: str,
) -> int:
    """Run the test files of `test_pairs` in one pytest run that records coverage contexts.

    The tests of each test file run on one pytest-xdist worker, or all in one process if `num_jobs`
    is 0.  The ``_coverage_contexts`` plugin labels the lines that run with coverage contexts, and
    reduces the state that the test files share (see that module).

    Returns:
        The exit code of pytest.
    """
    test_files = sorted({file for _, pair_test_files in test_pairs for file in pair_test_files})
    measured_files = sorted({file for files, _ in test_pairs for file in files})
    if num_jobs is None:
        num_jobs = min((os.cpu_count() or 2) // 2, len(test_files))
    command = _get_single_run_command(test_files, num_jobs, pytest_args)
    environment = _get_single_run_environment(measured_files, test_files, data_file, branch)
    return subprocess.call(command, cwd=check_utils.root_dir, env={**os.environ, **environment})


def _get_single_run_command(
    test_files: list[str], num_jobs: int, pytest_args: list[str]
) -> list[str]:
    """Return the pytest command that runs `test_files` with the plugin of coverage contexts."""
    # Pytest loads the conftest.py files of the paths that it is given before it collects tests, so
    # it is given only the directory that contains every test file, whose conftest.py files every
    # test file needs.  The plugin restricts collection to the test files.  The root directory of
    # pytest sets the paths in node IDs, which name the coverage contexts.
    common_directory = os.path.commonpath([os.path.dirname(file) for file in test_files])
    command = [sys.executable, "-m", "pytest", common_directory or os.curdir]
    command += [f"--rootdir={check_utils.root_dir}", f"-n={num_jobs}"]
    if num_jobs:
        command.append("--dist=loadfile")
    return [*command, "-p", "checks_superstaq._coverage_contexts", *pytest_args]


def _get_single_run_environment(
    measured_files: list[str], test_files: list[str], data_file: str, branch: bool
) -> dict[str, str]:
    """Return the environment variables that configure the plugin of coverage contexts."""
    root_dir = check_utils.root_dir
    return {
        protocol.DATA_FILE_ENV: data_file,
        protocol.ROOT_DIR_ENV: root_dir,
        protocol.MEASURED_FILES_ENV: os.pathsep.join(
            os.path.join(root_dir, file) for file in measured_files
        ),
        protocol.SELECTED_TEST_FILES_ENV: os.pathsep.join(
            os.path.join(root_dir, file) for file in test_files
        ),
        protocol.BRANCH_ENV: "1" if branch else "",
    }


def _report_single_run(test_pairs: _TestPairs, data_file: str) -> int:
    """Report the files of `test_pairs` whose coverage by their test files is below the threshold.

    The coverage of each pair is computed separately, from only the contexts credited to its test
    files (see `_credited_context_regexes`), and compared with ``fail_under`` in the coverage
    configuration.  As in a modular check with a process for each pair, files that are not
    measured, such as files that no test imports or that the configuration omits, are not reported.

    Returns:
        2 if the coverage of any file is below the threshold (as ``coverage report`` returns), and
        0 otherwise.
    """
    measurement = coverage.Coverage(data_file=data_file)
    measurement.combine(keep=False)
    measurement.save()
    measured_files = {
        protocol.normalize_path(file) for file in measurement.get_data().measured_files()
    }
    meets_threshold = True
    for files, test_files in test_pairs:
        measured = [file for file in files if protocol.normalize_path(file) in measured_files]
        source_files = [file for file in measured if file not in test_files]
        own_test_files = [file for file in measured if file in test_files]
        if source_files:
            meets_threshold &= _report_if_below_threshold(
                measurement, source_files, test_files, include_all_collection_contexts=False
            )
        if own_test_files:
            meets_threshold &= _report_if_below_threshold(
                measurement, own_test_files, test_files, include_all_collection_contexts=True
            )
    return 0 if meets_threshold else 2


def _report_if_below_threshold(
    measurement: coverage.Coverage,
    files: list[str],
    test_files: list[str],
    *,
    include_all_collection_contexts: bool,
) -> bool:
    """Print a coverage report of `files` if their coverage by `test_files` is below the threshold.

    Returns:
        Whether the coverage of `files` by the contexts of `test_files` meets the threshold.
    """
    report = io.StringIO()
    contexts = _credited_context_regexes(
        test_files, include_all_collection_contexts=include_all_collection_contexts
    )
    try:
        percentage = measurement.report(morfs=files, contexts=contexts, precision=2, file=report)
    except coverage.exceptions.NoDataError:  # the report configuration omits the files
        return True
    if percentage >= cast("float", measurement.get_option("report:fail_under")):
        return True
    heading = f"Coverage credited to {', '.join(test_files)} for {', '.join(files)}:"
    print(f"\n{heading}\n{report.getvalue()}", end="")  # noqa: T201
    return False


def _credited_context_regexes(
    test_files: list[str], *, include_all_collection_contexts: bool
) -> list[str]:
    """Return regular expressions that match the coverage contexts credited to `test_files`.

    These are the contexts of startup and of first imports after startup, which all test files
    share, the collection of `test_files` and of their parent directories, and the phases of their
    tests.  With `include_all_collection_contexts`, which is for reporting the coverage of test
    files themselves, they also include the collection by every collector: a test file's
    module-level lines run during the collection of whichever test file imports it first.
    """
    contexts = {protocol.STARTUP_CONTEXT, protocol.SHARED_IMPORT_CONTEXT}
    regexes = []
    if include_all_collection_contexts:
        regexes.append(f"^.*{re.escape(protocol.get_collection_context(''))}$")
    # The suffixes that the contexts of the phases of a test add to its node ID.
    test_suffixes = "|".join(
        re.escape(protocol.get_test_context("", phase)) for phase in protocol.TEST_PHASES
    )
    for test_file in test_files:
        # Pytest node IDs, which name the contexts, separate directories with "/" on every platform.
        path = pathlib.PurePath(test_file)
        node_ids = [parent.as_posix() for parent in path.parents if parent.parts]
        node_ids.append(path.as_posix())
        contexts.update(map(protocol.get_collection_context, node_ids))
        regexes.append(f"^{re.escape(path.as_posix())}::.*({test_suffixes})$")
    return [f"^({'|'.join(map(re.escape, sorted(contexts)))})$", *regexes]


def _run_on_files(
    files_requiring_coverage: list[str],
    test_files: list[str],
    coverage_args: list[str],
    pytest_args: list[str],
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Helper function to run coverage tests on the given files with the given pytest arguments."""
    coverage_args = ["--include=" + ",".join(files_requiring_coverage), *coverage_args]
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            *coverage_args,
            "-m",
            "pytest",
            *test_files,
            *pytest_args,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
    )


def _report(test_returncode: int) -> int:
    subprocess.call(
        [sys.executable, "-m", "coverage", "combine"],
        cwd=check_utils.root_dir,
        stdout=subprocess.DEVNULL,
    )

    # Flush Python's stdout buffer before the subprocess writes directly to it, so captured
    # test output appears before the coverage report
    sys.stdout.flush()
    coverage_returncode = subprocess.call(
        [sys.executable, "-m", "coverage", "report", "--precision=2"],
        cwd=check_utils.root_dir,
    )
    return _finish(test_returncode, coverage_returncode)


def _finish(test_returncode: int, coverage_returncode: int) -> int:
    """Print the result of a check, and return its exit code."""
    if test_returncode:
        print(check_utils.failure("TEST FAILURE!"))  # noqa: T201
        return test_returncode

    if coverage_returncode:
        print(check_utils.failure("COVERAGE FAILURE!"))  # noqa: T201
        return coverage_returncode

    print(check_utils.success("TEST AND COVERAGE SUCCESS!"))  # noqa: T201
    return 0


def _abort(error: BaseException) -> int:
    """Erase the coverage data of a check that did not complete, and report its error."""
    subprocess.check_call([sys.executable, "-m", "coverage", "erase"], cwd=check_utils.root_dir)
    print(check_utils.failure(f"\nCoverage failed to complete ({error!r})."))  # noqa: T201
    return 1


@contextlib.contextmanager
def _working_directory(directory: str) -> Iterator[None]:
    """Change the working directory temporarily."""
    previous_directory = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous_directory)


if __name__ == "__main__":
    sys.exit(run(*sys.argv[1:]))
