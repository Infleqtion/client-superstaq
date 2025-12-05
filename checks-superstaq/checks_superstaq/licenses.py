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

import subprocess
import sys
import textwrap
from collections.abc import Iterable

from checks_superstaq import check_utils


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: str | Iterable[str] = "*.py",
    exclude: str | Iterable[str] = "*.ipynb",
    silent: bool = False,
) -> int:
    """Runs the 'licensepy format' tool on the repository (license header check/formatting).

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
        Runs 'licensepy format' on the repository (adds license headers to source code)
        """
    )
    parser.add_argument("--fix", action="store_true", help="Apply changes to files.")

    parsed_args, args_to_pass = parser.parse_known_intermixed_args(args)

    if "licenses" in parsed_args.skip:
        return 0

    if not parsed_args.fix:
        args_to_pass.append("--dry-run")

    files = check_utils.extract_files(parsed_args, include, exclude, silent)

    if files:
        return_code = subprocess.call(
            ["licensepy", "format", *files, *args_to_pass],
            cwd=check_utils.root_dir,
        )
        if return_code > 0 and not parsed_args.fix:
            command = "./checks/licenses.py --fix"
            text = (
                f"Run '{command}' (from the repo root directory) to format "
                "files with correct license header."
            )
            print(check_utils.warning(text))  # noqa: T201
            return return_code

        return return_code

    return 0


if __name__ == "__main__":
    sys.exit(run(*sys.argv[1:]))
