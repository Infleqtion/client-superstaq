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

import fnmatch
import functools
import importlib.metadata
import json
import os
import re
import sys
import textwrap
import urllib.request
from collections.abc import Iterable

import packaging.version

from checks_superstaq import check_utils


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: str | Iterable[str] = "*requirements.txt",
    exclude: str | Iterable[str] = "",
    upstream_match: str = "*superstaq*",
    silent: bool = False,
) -> int:
    """Checks that:
    - all pip requirements files (i.e. files matching *requirements.txt) are sorted
    - all upstream packages are pinned to their latest versions.

    Args:
        *args: Command line arguments.
        include: Glob(s) indicating which tracked files to consider (e.g. "*.py").
        exclude: Glob(s) indicating which tracked files to skip (e.g. "*integration_test.py").
        upstream_match: String to match package name and version.
        silent: If True, restrict printing to warning and error messages.

    Returns:
        Terminal exit code. 0 indicates success, while any other integer indicates a test failure.
    """
    parser = check_utils.get_check_parser()
    parser.description = textwrap.dedent(
        """
        Checks that:
          - all pip requirements files (i.e. files matching *requirements.txt) are sorted
          - all upstream packages are pinned to their latest versions
        Also throws a warning if the local version of upstream packages are out of date.

        This script exits with a succeeding exit code if and only if all pip requirements files
        are properly formatted.
        """
    )

    parser.add_argument(
        "--fix", "--apply", action="store_true", help="Apply fixes to pip requirements files."
    )
    parser.add_argument(
        "--only-sort",
        action="store_true",
        help="Only sort requirements files.  Do not check upstream package versions.",
    )
    parsed_args = parser.parse_intermixed_args(args)
    if "requirements" in parsed_args.skip:
        return 0

    files = check_utils.extract_files(parsed_args, include, exclude, silent)

    # check all requirements files
    requirements_to_fix = {}
    for req_file in files:
        needs_cleanup, requirements = _inspect_req_file(
            req_file, parsed_args.only_sort, upstream_match, silent
        )
        if needs_cleanup:
            requirements_to_fix[req_file] = requirements

    # print some helpful text and maybe apply fixes
    _cleanup(requirements_to_fix, parsed_args.fix, silent)

    success = not requirements_to_fix or parsed_args.fix
    return 0 if success else 1


def _inspect_req_file(
    req_file: str, only_sort: bool, upstream_match: str, silent: bool
) -> tuple[bool, list[str]]:
    # read in requirements line-by-line
    with open(os.path.join(check_utils.root_dir, req_file)) as file:
        requirements = file.read().strip().split("\n")

    if not _are_pip_requirements(requirements):
        error = f"{req_file} appears to contain lines that are not valid pip requirements"
        if req_file == "requirements.txt":
            raise SyntaxError(check_utils.failure(error))
        elif not silent:
            print(check_utils.warning(error))  # noqa: T201
        return False, []  # file cannot be cleaned up, and there are no requirements to track

    needs_cleanup, requirements = _sort_requirements(requirements)
    if needs_cleanup and not silent:
        print(check_utils.failure(f"{req_file} is not sorted."))  # noqa: T201

    if not only_sort:
        needs_cleanup |= _check_package_versions(
            req_file, requirements, upstream_match, silent, strict=True
        )

    return needs_cleanup, requirements


def _are_pip_requirements(requirements: list[str]) -> bool:
    # "official" regex for pypi package names:
    # https://peps.python.org/pep-0508/#names
    pypi_name = r"[A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9]"
    extras = rf"{pypi_name}(\s*,\s*{pypi_name})*"  # comma(+whitespace)-delineated list

    # regex for version compatibility specification:
    # https://peps.python.org/pep-0440/#version-specifiers
    relation = r"(==|~=|>|>=|<|<=|!=|===)"
    version = r"""
        [0-9]+         # at least one numeric character
        (\.[0-9]+)*    # repeat 0+ times: .numeric
        (\.[A-Z0-9]+)? # maybe once: .alphanumeric
        (\.\*)?        # maybe once: .* (literal)
    """
    restriction = rf"{relation}\s*{version}"  # relation, maybe whitespace, and version
    restrictions = rf"{restriction}(\s*,\s*{restriction})*"  # comma(+whitespace)-delineated list

    comment = r"\# .*"  # literal "#" followed by anything
    pip_req_format = re.compile(
        rf"^{pypi_name} (\[{extras}\])? \s* ({restrictions})? \s* ({comment})?",
        flags=re.IGNORECASE | re.VERBOSE,
    )
    return all(pip_req_format.match(requirement) for requirement in requirements)


def _get_package_name(requirement: str) -> str:
    return re.split(r"[><~=@]", requirement)[0]


def _sort_requirements(requirements: list[str]) -> tuple[bool, list[str]]:
    sorted_requirements = sorted(requirements, key=lambda req: _get_package_name(req).lower())
    needs_cleanup = requirements != sorted_requirements
    return needs_cleanup, sorted_requirements


def _check_package_versions(
    req_file: str, requirements: list[str], match: str, silent: bool, strict: bool
) -> bool:
    """Check whether package requirements matching 'match' are up-to-date with their latest
    versions.
    Print warnings if matching requirements are out of date.  Return whether the requirements file
    *must* be updated, i.e., return 'True' iff packages are out of date and 'strict == True'.
    """
    text_format = check_utils.failure if strict else check_utils.warning

    up_to_date = True
    for idx, req in enumerate(requirements):
        package = _get_package_name(req)
        if not fnmatch.fnmatch(package, match):
            # this is not an upstream package
            continue

        latest_version = _get_latest_version(package, silent)
        desired_req = f"{package}~={latest_version}"
        latest_version_is_required = desired_req == req
        up_to_date &= bool(latest_version_is_required)

        if strict and not latest_version_is_required:
            requirements[idx] = desired_req

        if not silent:
            if not latest_version_is_required:
                should_require = desired_req.lstrip(package)
                pin_text = f"{req_file} requires {req}, but should require {should_require}"
                print(text_format(pin_text))  # noqa: T201

            # check that a compatible version of this upstream package is installed locally
            _inspect_local_version(package, latest_version)

    return strict and not up_to_date  # True iff we are *requiring* changes to the requirements file


@functools.lru_cache
def _get_latest_version(package: str, silent: bool) -> str:
    """Retrieve the latest version of a package."""
    local_version = _get_local_version(package)
    pypi_version = _get_pypi_version(package, silent)
    if not local_version and not pypi_version:
        raise ModuleNotFoundError(f"Package not installed or found on PyPI: {package}")
    return max(pypi_version or "0.0.0", local_version or "0.0.0", key=packaging.version.parse)


def _get_local_version(package: str) -> str | None:
    """Retrieve the local version of a package (if installed)."""
    base_package = package.split("[")[0]  # remove options: package_name[options] --> package_name
    sanitized_package_name = base_package.replace("-", "_").lower()
    try:
        module = importlib.import_module(sanitized_package_name)
        if hasattr(module, "__version__"):
            return module.__version__
        return importlib.metadata.version(sanitized_package_name)
    except ModuleNotFoundError:
        return None


def _get_pypi_version(package: str, silent: bool) -> str | None:
    """Retrieve the latest version of a package on PyPI (if found)."""
    base_package = package.split("[")[0]  # remove options: package_name[options] --> package_name
    pypi_url = f"https://pypi.org/pypi/{base_package}/json"
    try:
        package_info = urllib.request.urlopen(pypi_url).read().decode()
        pypi_version = json.loads(package_info)["info"]["version"]
    except urllib.error.URLError:
        if not silent:
            warning = f"Cannot find package on PyPI: {base_package}."
            print(check_utils.warning(warning))  # noqa: T201
        return None
    else:
        return pypi_version


def _inspect_local_version(package: str, latest_version: str) -> None:
    """Check that the package is installed with the same minor version (X.Y.*) as latest_version."""
    local_version = _get_local_version(package)
    if not local_version or local_version.split(".")[:2] != latest_version.split(".")[:2]:
        warning = f"WARNING: locally installed version of {package} is not up to date."
        version_info = f"Local version is {local_version}, but latest version is {latest_version}."
        suggestion = f"Try calling 'python -m pip install --upgrade {package}'."
        print(check_utils.warning(warning))  # noqa: T201
        print(check_utils.warning(version_info))  # noqa: T201
        print(check_utils.warning(suggestion))  # noqa: T201


def _cleanup(
    requirements_to_fix: dict[str, list[str]],
    apply_changes: bool,
    silent: bool,
) -> None:
    if not requirements_to_fix:
        print("Nothing to fix in requirements files.")  # noqa: T201

    elif apply_changes:
        for req_file, requirements in requirements_to_fix.items():
            with open(os.path.join(check_utils.root_dir, req_file), "w") as file:
                file.write("\n".join(requirements) + "\n")
        if not silent:
            print(check_utils.success("Requirements files fixed."))  # noqa: T201

    elif not silent:
        command = "./checks/requirements.py --fix"
        text = f"Run '{command}' (from the repo root directory) to fix requirements files."
        print(check_utils.warning(text))  # noqa: T201


if __name__ == "__main__":
    sys.exit(run(*sys.argv[1:]))
