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

import importlib.util
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterable

from checks_superstaq import check_utils

os.environ["FORCE_COLOR"] = "1"


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: str | Iterable[str] = ("*.py", "*.ipynb"),
    include_pyproject: str | Iterable[str] = "*pyproject.toml",
    exclude: str | Iterable[str] = (),
    silent: bool = False,
) -> int:
    """Runs 'ruff format' and 'pyproject-fmt' on the repository (formatting check).

    Args:
        *args: Command line arguments.
        include: Glob(s) indicating which tracked files to format with 'ruff format' (e.g. "*.py").
        include_pyproject: Glob(s) indicating which tracked files to format with 'pyproject-fmt'.
        exclude: Glob(s) indicating which tracked files to skip (e.g. "*integration_test.py").
        silent: If True, restrict printing to warning and error messages.

    Returns:
        Terminal exit code. 0 indicates success, while any other integer indicates a test failure.
    """
    parser = check_utils.get_check_parser()
    parser.description = textwrap.dedent(
        """
        Runs 'ruff format' on python files and notebooks, and 'pyproject-fmt' on pyproject.toml
        files (formatting check).
        """
    )

    parser.add_argument("--fix", action="store_true", help="Apply changes to files.")

    parsed_args, args_to_pass = parser.parse_known_intermixed_args(args)
    if "format" in parsed_args.skip:
        return 0

    if not parsed_args.fix:
        args_to_pass.append("--diff")

    # Identify files for ruff
    files = check_utils.extract_files(parsed_args, include, exclude, silent)

    # Identify files for pyproject-fmt
    pyproject_files = []
    include_pyproject = (
        [include_pyproject] if isinstance(include_pyproject, str) else list(include_pyproject)
    )
    if include_pyproject:
        pyproject_files = check_utils.extract_files(
            parsed_args, include_pyproject, exclude, silent=True
        )
        # Files passed directly as arguments are always extracted, so filter them again here
        pyproject_files = check_utils.select_files(pyproject_files, include_pyproject)

    returncode = 0
    if files:
        returncode = subprocess.call(
            [sys.executable, "-m", "ruff", "format", *files, *args_to_pass],
            cwd=check_utils.root_dir,
        )
    if pyproject_files:
        # Avoid formatting the same file twice (e.g. via a symlink to another tracked file)
        real_paths = {
            os.path.realpath(os.path.join(check_utils.root_dir, file)): file
            for file in reversed(pyproject_files)
        }
        pyproject_files = sorted(real_paths.values())
        returncode = max(returncode, _run_pyproject_fmt(pyproject_files, fix=parsed_args.fix))

    if returncode == 1:
        command = "./checks/format_.py --fix"
        text = f"Run '{command}' (from the repo root directory) to format files."
        print(check_utils.warning(text))  # noqa: T201

    return returncode


def _run_pyproject_fmt(files: list[str], *, fix: bool) -> int:
    """Runs 'pyproject-fmt' on the given files.

    Args:
        files: The pyproject.toml files to check or format.
        fix: If True, format files in place. Otherwise only report formatting issues.

    Returns:
        Terminal exit code. 0 indicates that all files are (now) formatted correctly.
    """
    if importlib.util.find_spec("pyproject_fmt") is None:
        text = (
            "Skipping pyproject.toml formatting because 'pyproject-fmt' is not installed "
            "(it requires Python 3.10 or later)."
        )
        print(check_utils.warning(text))  # noqa: T201
        return 0

    # Ignore runpy's (benign) RuntimeWarning about pyproject_fmt.__main__ being imported twice
    command = [sys.executable, "-W", "ignore::RuntimeWarning:runpy", "-m", "pyproject_fmt"]

    if fix:
        # pyproject-fmt exits with 1 whenever it changes a file, so re-check to get the final status
        subprocess.call([*command, *files], cwd=check_utils.root_dir)
        return subprocess.call(
            [*command, "--check", "--no-print-diff", *files], cwd=check_utils.root_dir
        )

    return subprocess.call([*command, "--check", *files], cwd=check_utils.root_dir)


if __name__ == "__main__":
    sys.exit(run(*sys.argv[1:]))
