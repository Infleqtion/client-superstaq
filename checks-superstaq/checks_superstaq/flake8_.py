#!/usr/bin/env python3
# Copyright 2024 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
    exclude: str | Iterable[str] = (),
    silent: bool = False,
) -> int:
    """Runs flake8 on the repository (formatting check).

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
        Runs flake8 on the repository (formatting check).
        """
    )

    parsed_args, args_to_pass = parser.parse_known_intermixed_args(args)
    if "flake8" in parsed_args.skip:
        return 0

    files = check_utils.extract_files(parsed_args, include, exclude, silent)

    if files:
        return subprocess.call(
            ["python", "-m", "flake8", *files, *args_to_pass], cwd=check_utils.root_dir
        )

    return 0


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
