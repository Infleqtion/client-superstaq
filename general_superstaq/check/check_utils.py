#!/usr/bin/env python3

"""
Dumping ground for check script utilities.
"""

import argparse
import dataclasses
import fnmatch
import os
import re
import subprocess
import sys
from typing import Any, Callable, Iterable, List, Optional, Union

# identify the root directory of the "main" script that called this module
main_file_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
root_dir = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], cwd=main_file_dir, text=True
).strip()


def _check_output(*commands: str) -> str:
    """Wrapper for subprocess.check_output to run commands from root_dir and clean up the output."""
    return subprocess.check_output(commands, text=True, cwd=root_dir).strip()


# container for string formatting console codes
@dataclasses.dataclass
class Style:
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def styled(text: str, style_code: str) -> str:
    return style_code + text + Style.RESET


def warning(text: str) -> str:
    return styled(text, Style.RED)


def failure(text: str) -> str:
    return styled(text, Style.BOLD + Style.RED)


def success(text: str) -> str:
    return styled(text, Style.BOLD + Style.GREEN)


# default branches to compare against when performing incremental checks
default_branches = ("upstream/main", "origin/main", "main")


####################################################################################################
# methods for identifying files to check


def get_tracked_files(
    include: Union[str, Iterable[str]],
    exclude: Union[str, Iterable[str]] = "",
) -> List[str]:
    """
    Identify all files matching the given match_patterns that are tracked by git in this repo.
    If no matches are provided, return a list of all python scripts in the repo.

    Optionally excludes anything that matches [root_dir]/exclusion for each given exclusion (passed
    either as a single string or a list of strings).
    """
    match_patterns = [include] if isinstance(include, str) else list(include)
    matching_files = _check_output("git", "ls-files", *match_patterns).splitlines()
    should_include = inclusion_filter(exclude)
    return [file for file in matching_files if should_include(file)]


def inclusion_filter(exclude: Union[str, Iterable[str]]) -> Callable[[str], bool]:
    """Construct filter that decides whether a file should be included."""
    if not exclude:
        return lambda _: True

    exclusions = [exclude] if isinstance(exclude, str) else exclude

    def should_include(file: str) -> bool:
        return not any(fnmatch.fnmatch(file, exclusion) for exclusion in exclusions)

    return should_include


def get_changed_files(
    include: Union[str, Iterable[str]],
    exclude: Union[str, Iterable[str]],
    revisions: Optional[Iterable[str]] = None,
    silent: bool = False,
) -> List[str]:
    """
    Get the files of interest that have been changed in the current branch.
    Here "files of interest" means all files identified by get_tracked_files (see above).

    You can specify git revisions to compare against when determining whether a file is considered
    to have "changed".  If multiple revisions are provided, this script compares against their most
    recent common ancestor.

    If an empty list of revisions is specified, this script will default to the first of the
    default_branches (specified above) that it finds.  If none of these branches exists, this method
    raises a ValueError.
    """
    if revisions is None:
        return []
    else:
        revisions = list(revisions)

    # verify that all arguments are valid revisions
    invalid_revisions = [revision for revision in revisions if not _revision_exists(revision)]
    if invalid_revisions:
        rev_text = " ".join([f"'{rev}'" for rev in invalid_revisions])
        raise ValueError(failure(f"Revision(s) not found: {rev_text}"))

    # identify the revision to compare against
    base_revision = _get_ancestor(*revisions, silent=silent)

    # identify the revision to diff against ~ most recent common ancestor of base_revision and HEAD
    common_ancestor = _get_ancestor(base_revision, "HEAD", silent=True)
    if not silent:
        revision_commit = _check_output("git", "rev-parse", base_revision)
        if common_ancestor == revision_commit:
            print(f"Comparing against revision '{base_revision}'")
        else:
            print(f"Comparing against revision '{base_revision}' (merge base '{common_ancestor}')")

    changed_files = _check_output("git", "diff", "--name-only", common_ancestor).splitlines()
    files_to_examine = [
        file for file in get_tracked_files(include, exclude) if file in changed_files
    ]

    if not silent:
        print(f"Found {len(files_to_examine)} changed file(s) to examine")
        for file in files_to_examine:
            print(file)
    return files_to_examine


def _get_ancestor(*revisions: str, silent: bool = False) -> str:
    """
    Helper function to identify the most recent common ancestor of the given git revisions.
    """
    if len(revisions) == 1:
        return revisions[0]

    elif len(revisions) > 1:
        if not silent:
            rev_text = " ".join([f"'{rev}'" for rev in revisions])
            print(f"Finding common ancestor of revisions {rev_text}")
        return _check_output("git", "merge-base", *revisions)

    else:
        for branch in default_branches:
            if _revision_exists(branch):
                return branch
        error = f"Default git revisions not found: {default_branches}"
        raise RuntimeError(failure(error))


def _revision_exists(revision: str) -> bool:
    """Helper function to check whether a git revision exists."""
    return not subprocess.call(
        ["git", "rev-parse", "--verify", revision],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=root_dir,
    )


def get_test_files(*files: str, exclude: Union[str, Iterable[str]] = "", silent: bool) -> List[str]:
    """
    For the given files, identify all associated test files (i.e. files with the same name, but
    with a "_test.py" suffix).
    """
    should_include = inclusion_filter(exclude)

    test_files = set()
    for file in files:
        if file.endswith("_test.py"):
            test_files.add(file)

        else:
            test_file = re.sub(r"\.py$", "_test.py", file)
            test_file_exists = os.path.isfile(os.path.join(root_dir, test_file))
            if test_file_exists and should_include(test_file):
                test_files.add(test_file)
            elif not silent:
                print(warning(f"WARNING: no test file found for {file}"))

    return list(test_files)


####################################################################################################
# file parsing, incremental checks, and decorator to exit instead of returning a failing exit code


def get_file_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    help_text = "The files to check. If not passed any files, inspects the entire repo."
    parser.add_argument("files", nargs="*", help=help_text)

    help_text = (
        "Run an incremental check on files that have changed since a specified revision.  "
        + f"If no revisions are specified, compare against the first of {default_branches} "
        + "that exists.  If multiple revisions are provided, this script compares against "
        + "their most recent common ancestor.  Incremental checks ignore integration tests."
    )
    parser.add_argument("-i", "--incremental", dest="revisions", nargs="*", help=help_text)

    return parser


def extract_files(
    parsed_args: argparse.Namespace,
    include: Union[str, Iterable[str]],
    exclude: Union[str, Iterable[str]] = "",
    silent: bool = False,
    search_if_empty: bool = True,
) -> List[str]:
    files = parsed_args.files if "files" in parsed_args else []
    if "revisions" in parsed_args:
        files += get_changed_files(include, exclude, parsed_args.revisions, silent=silent)
    if not files and search_if_empty:
        return get_tracked_files(include, exclude)
    else:
        return files


def enable_exit_on_failure(func_with_returncode: Callable[..., int]) -> Callable[..., int]:
    """
    Decorator optionally allowing a function to exit instead of returning a failing return code.
    """

    def func_with_exit(*args: Any, exit_on_failure: bool = False, **kwargs: Any) -> int:
        returncode = func_with_returncode(*args, **kwargs)
        if exit_on_failure and returncode:
            exit(returncode)
        return returncode

    return func_with_exit
