#!/usr/bin/env python3

import argparse
import functools
import glob
import json
import os
import re
import subprocess
import sys
import textwrap
import urllib.request
from typing import Dict, Iterable, List, Optional, Union

import pkg_resources

from applications_superstaq.check import check_utils


@check_utils.enable_exit_on_failure
@check_utils.extract_file_args
def run(
    *args: str,
    files: Optional[Iterable[str]] = None,
    parser: argparse.ArgumentParser = check_utils.get_file_parser(add_files=False),
    exclude: Optional[Union[str, Iterable[str]]] = None,
    upstream_match: str = ".*superstaq",
    silent: bool = False,
    only_sort: bool = False,
) -> int:

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
        "--apply", action="store_true", help="Apply fixes to pip requirements files."
    )
    parser.add_argument(
        "--only-sort",
        action="store_true",
        help="Only sort requirements files.  Do not check upstream package versions.",
    )
    parsed_args = parser.parse_args(args)
    only_sort |= parsed_args.only_sort

    if files is None:
        req_file_match = os.path.join(check_utils.root_dir, "**", "*requirements.txt")
        files = [
            os.path.relpath(file, check_utils.root_dir)
            for file in glob.iglob(req_file_match, recursive=True)
        ]
    files = filter(check_utils.inclusion_filter(exclude), files)

    requirements_to_fix = {}
    for req_file in files:

        with open(os.path.join(check_utils.root_dir, req_file), "r") as file:
            requirements = file.read().strip().split("\n")

        if not _are_pip_requirements(requirements):
            error = f"{req_file} not recognized as a pip requirements file."
            if req_file == "requirements.txt":
                raise SyntaxError(check_utils.failure(error))
            elif not silent:
                print(check_utils.warning(error))
            continue

        is_tidy = _sort_requirements(requirements)
        if not is_tidy and not silent:
            print(check_utils.failure(f"{req_file} is not sorted."))

        if not only_sort:
            is_tidy &= _pin_upstream_packages(requirements, upstream_match, silent)

        if not is_tidy:
            requirements_to_fix[req_file] = requirements

        # check whether all requirements for this repo are satisfied
        if req_file == "requirements.txt" and not silent:
            _check_requirements(requirements)

    # print some helpful text and maybe apply fixes
    _cleanup(requirements_to_fix, parsed_args.apply, silent)

    success = not requirements_to_fix or parsed_args.apply
    return 0 if success else 1


def _are_pip_requirements(requirements: List[str]) -> bool:
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


def _sort_requirements(requirements: List[str]) -> bool:
    sorted_requirements = sorted(requirements, key=str.casefold)
    is_sorted = requirements == sorted_requirements
    if not is_sorted:
        requirements[:] = sorted_requirements
    return is_sorted


def _pin_upstream_packages(requirements: List[str], upstream_match: str, silent: bool) -> bool:
    # identify upstream package versions
    upstream_versions = {
        package: _get_latest_version(package)
        for requirement in requirements
        if re.match(upstream_match, package := re.split(">|<|~|=", requirement)[0])
    }

    # pin upstream packages to their latest versions
    up_to_date = True
    for package, latest_version in upstream_versions.items():
        requirement = f"{package}=={latest_version}"
        package_pinned = requirement in requirements
        up_to_date &= package_pinned

        if not package_pinned:
            if not silent:
                print(check_utils.failure(f"{requirement} not pinned in requirements.txt."))

            # update requirements file contents
            for idx, old_req in enumerate(requirements):
                if package == re.split(">|<|~|=", old_req)[0]:
                    requirements[idx] = requirement
                    break

            if not silent:
                # print warning if the wrong version of an upstream package is installed locally
                _inspect_local_version(package, latest_version)

    return up_to_date


@functools.lru_cache
def _get_latest_version(package: str) -> str:
    pypi_url = f"https://pypi.org/pypi/{package}/json"
    return json.loads(urllib.request.urlopen(pypi_url).read().decode())["info"]["version"]


def _inspect_local_version(package: str, latest_version: str) -> None:
    try:
        installed_info = subprocess.check_output(
            [sys.executable, "-m", "pip", "show", package],
            cwd=check_utils.root_dir,
            text=True,
        ).split()
        installed_version = installed_info[installed_info.index("Version:") + 1]
        assert installed_version == latest_version
    except (subprocess.CalledProcessError, ValueError, AssertionError):
        warning = f"WARNING: {package} is not up to date."
        suggestion = f"Try calling 'python -m pip install --upgrade {package}'."
        print(check_utils.warning(warning))
        print(check_utils.warning(suggestion))


def _check_requirements(requirements: List[str]) -> None:
    # wrapper for "pkg_resources.require" to check that all requirements are satisfied
    try:
        pkg_resources.require(iter(requirements))
    except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict) as error:
        print(check_utils.warning("WARNING: " + error.report()))


def _cleanup(
    requirements_to_fix: Dict[str, List[str]],
    apply_changes: bool,
    silent: bool,
) -> None:
    if not requirements_to_fix:
        print("Nothing to fix in requirements files.")

    else:
        if apply_changes:
            for req_file, requirements in requirements_to_fix.items():
                with open(os.path.join(check_utils.root_dir, req_file), "w") as file:
                    file.write("\n".join(requirements) + "\n")
            if not silent:
                print("Requirements files fixed.")

        elif not silent:
            this_file = os.path.relpath(__file__)
            print(f"Run '{this_file} --apply' to fix requirements files.")


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
