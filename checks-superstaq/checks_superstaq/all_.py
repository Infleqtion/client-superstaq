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
import sys
import textwrap
import warnings

from checks_superstaq import (
    build_docs,
    check_utils,
    configs,
    coverage_,
    format_,
    licenses,
    lint_,
    mypy_,
    requirements,
)


def run(*args: str) -> int:
    """Runs all checks on the repository.

    Args:
        *args: Command line arguments.

    Returns:
        Terminal exit code. 0 indicates success, while any other integer indicates a test failure.
    """
    parser = check_utils.get_check_parser()
    parser.description = textwrap.dedent(
        """
        Runs all checks on the repository.
        Exits immediately upon any failure unless passed one of -f, -F, or --force as an argument.
        This script exits with a succeeding exit code if and only if all checks pass.
        """
    )

    parser.add_argument(
        "-f",
        "--force-formats",
        action="store_true",
        help="'Soft force' ~ continue past (i.e. do not exit after) failing format checks.",
    )
    parser.add_argument(
        "-F",
        "--force",
        dest="force_all",
        action="store_true",
        help="'Hard force' ~ continue past (i.e. do not exit after) all failing checks.",
    )
    parser.add_argument(
        "--ruff",
        action="store_true",
        help="[DEPRECATED] Ruff is now the default formatter and linter. "
        "The --ruff flag is ignored.",
    )
    parser.add_argument(
        "--sysmon",
        action="store_true",
        help="Enable the `COVERAGE_CORE=sysmon` env variable for faster coverage (requires "
        "Python 3.12 or higher).",
    )
    parser.add_argument(
        "--licensepy",
        action="store_true",
        help="Enable running the 'licensepy format' tool on the repository (license header "
        "check/formatting)",
    )

    parsed_args, _ = parser.parse_known_intermixed_args(args)

    # Issue deprecation warning for `--ruff` flag
    if parsed_args.ruff:
        warnings.warn(
            "The --ruff flag is deprecated and will be removed in a future version. "
            "Ruff is now the default formatter and linter.",
            DeprecationWarning,
            stacklevel=2,
        )

    if parsed_args.revisions is not None:
        # Print info about incremental files once now, rather than in each check
        _ = check_utils.extract_files(parsed_args, silent=False)

    if parsed_args.sysmon and sys.version_info.minor >= 12:
        os.environ["COVERAGE_CORE"] = "sysmon"

    default_mode = not parsed_args.files and parsed_args.revisions is None
    checks_failed = 0

    args_to_pass = [
        arg
        for arg in args
        if arg
        not in ("-f", "--force-formats", "-F", "--force", "--ruff", "--sysmon", "--licensepy")
    ]

    # Always use ruff for formatting and linting (remove the conditional logic)
    # run formatting checks
    # silence most checks to avoid printing duplicate info about incremental files
    # silencing does not affect warnings and errors
    exit_on_failure = not (parsed_args.force_formats or parsed_args.force_all)
    checks_failed |= configs.run(*args_to_pass, exit_on_failure=exit_on_failure, silent=True)
    checks_failed |= format_.run(*args_to_pass, exit_on_failure=exit_on_failure, silent=True)
    checks_failed |= lint_.run(*args_to_pass, exit_on_failure=exit_on_failure, silent=True)

    # Run license header checking (if enabled)
    if parsed_args.licensepy:
        checks_failed |= licenses.run(
            *args_to_pass,
            exit_on_failure=exit_on_failure,
            silent=True,
        )

    # Run typing and coverage checks
    exit_on_failure = not parsed_args.force_all
    checks_failed |= mypy_.run(*args_to_pass, exit_on_failure=exit_on_failure, silent=True)
    checks_failed |= coverage_.run(*args_to_pass, exit_on_failure=exit_on_failure, silent=True)

    # Check that all pip requirements files are in order
    checks_failed |= requirements.run(*args_to_pass, exit_on_failure=exit_on_failure)

    if default_mode:
        # Checks that the docs build
        checks_failed |= build_docs.run(
            *args_to_pass,
            exit_on_failure=exit_on_failure,
        )

    return checks_failed


if __name__ == "__main__":
    sys.exit(run(*sys.argv[1:]))
