#!/usr/bin/env python3

import fnmatch
import functools
import json
import os
import re
import subprocess
import sys
import textwrap
import urllib.request
from typing import Dict, Iterable, List, Tuple, Union

import pkg_resources

from applications_superstaq.check import check_utils


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: Union[str, Iterable[str]] = "*requirements.txt",
    exclude: Union[str, Iterable[str]] = "",
    upstream_match: str = "*superstaq*",
    silent: bool = False,
) -> int:

    parser = check_utils.get_file_parser()
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
    parsed_args = parser.parse_intermixed_args(args)
    files = check_utils.extract_files(parsed_args, include, exclude, silent)

    # check that we can connect to PyPI
    can_connect_to_pypi = _check_pypy_connection(silent)

    # check all requirements files
    requirements_to_fix = {}
    for req_file in files:
        needs_cleanup, requirements = _inspect_req_file(
            req_file, parsed_args.only_sort, can_connect_to_pypi, upstream_match, silent
        )
        if needs_cleanup:
            requirements_to_fix[req_file] = requirements

    # print some helpful text and maybe apply fixes
    _cleanup(requirements_to_fix, parsed_args.apply, silent)

    success = not requirements_to_fix or parsed_args.apply
    return 0 if success else 1


def _check_pypy_connection(silent: bool) -> bool:
    try:
        urllib.request.urlopen("https://pypi.org/", timeout=1)
        return True
    except urllib.error.URLError:
        if not silent:
            warning = "Cannot connect to PiPI to identify package versions to pin."
            print(check_utils.warning(warning))
        return False


def _inspect_req_file(
    req_file: str, only_sort: bool, can_connect_to_pypi: bool, upstream_match: str, silent: bool
) -> Tuple[bool, List[str]]:
    # read in requirements line-by-line
    with open(os.path.join(check_utils.root_dir, req_file), "r") as file:
        requirements = file.read().strip().split("\n")

    if not _are_pip_requirements(requirements):
        error = f"{req_file} not recognized as a pip requirements file."
        if req_file == "requirements.txt":
            raise SyntaxError(check_utils.failure(error))
        elif not silent:
            print(check_utils.warning(error))
        return False, []  # file cannot be cleaned up, and there are no requirements to track

    needs_cleanup, requirements = _sort_requirements(requirements)
    if needs_cleanup and not silent:
        print(check_utils.failure(f"{req_file} is not sorted."))

    if not only_sort and can_connect_to_pypi:
        needs_cleanup |= _pin_upstream_packages(requirements, upstream_match, silent)

    return needs_cleanup, requirements


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


def _sort_requirements(requirements: List[str]) -> Tuple[bool, List[str]]:
    sorted_requirements = sorted(requirements, key=str.casefold)
    needs_cleanup = requirements != sorted_requirements
    return needs_cleanup, sorted_requirements


def _pin_upstream_packages(requirements: List[str], upstream_match: str, silent: bool) -> bool:
    # identify upstream package versions
    upstream_versions = {
        package: _get_latest_version(package)
        for requirement in requirements
        if fnmatch.fnmatch(package := re.split(">|<|~|=", requirement)[0], upstream_match)
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

    return not up_to_date


@functools.lru_cache
def _get_latest_version(package: str) -> str:
    # remove options from package string, if present: package_name[options] --> package_name
    base_package = package.split("[")[0]
    pypi_url = f"https://pypi.org/pypi/{base_package}/json"
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
            print("Run 'check/requirements.py --apply' to fix requirements files.")


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
