#!/usr/bin/env python3

"""
Dumping ground for check script utilities.
"""

import argparse
import dataclasses
import fnmatch
import inspect
import os
import subprocess
import sys
from typing import Any, Callable, Iterable, List, Optional, Union

# identify the root directory of the git repository containing sys.argv[0]
_main_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
root_dir = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], cwd=_main_dir, text=True
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


def warning(text: str) -> str:
    return Style.RED + text + Style.RESET


def failure(text: str) -> str:
    return Style.BOLD + Style.RED + text + Style.RESET


def success(text: str) -> str:
    return Style.BOLD + Style.GREEN + text + Style.RESET


# default branches to compare against when performing incremental checks
default_branches = ("upstream/main", "origin/main", "main")


####################################################################################################
# methods for identifying files to check


def get_tracked_files(
    *match_patterns: str, exclude: Optional[Union[str, Iterable[str]]] = None
) -> List[str]:
    """
    Identify all files matching the given match_patterns that are tracked by git in this repo.
    If no matches are provided, return a list of all python scripts in the repo.

    Optionally excludes anything that matches [root_dir]/exclusion for each given exclusion (passed
    either as a single string or a list of strings).
    """
    matching_files = _check_output("git", "ls-files", *match_patterns).splitlines()
    should_include = inclusion_filter(exclude)
    return [file for file in matching_files if should_include(file)]


def inclusion_filter(exclude: Optional[Union[str, Iterable[str]]]) -> Callable[[str], bool]:
    """Construct filter that decides whether a file should be included."""
    if not exclude:
        return lambda _: True

    exclusions = [exclude] if isinstance(exclude, str) else exclude

    def should_include(file: str) -> bool:
        return not any(fnmatch.fnmatch(file, exclusion) for exclusion in exclusions)

    return should_include


def get_changed_files(
    match_patterns: Iterable[str],
    revisions: Iterable[str],
    silent: bool = False,
    exclude: Optional[Union[str, Iterable[str]]] = None,
) -> List[str]:
    """
    Get the files of interest that have been changed in the current branch.
    Here "files of interest" means all files identified by get_tracked_files (see above).

    You can optionally specify a git revisions to compare against when determining whether a file is
    considered to have "changed".  If multiple revisions are provided, this script compares against
    their most recent common ancestor.  If no revisions are specified, this script will default to
    the first of the default_branches (specified above) that it finds.  If none of these branches
    exists, this method raises a ValueError.
    """
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
    changed_and_included_files = list(filter(inclusion_filter(exclude), changed_files))
    files_to_examine = [
        file for file in get_tracked_files(*match_patterns) if file in changed_and_included_files
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

        error = f"No default git revision found to compare against {default_branches}"
        raise RuntimeError(failure(error))


def _revision_exists(revision: str) -> bool:
    """Helper function to check whether a git revision exists."""
    return not subprocess.call(
        ["git", "rev-parse", "--verify", revision],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=root_dir,
    )


####################################################################################################
# decorators to add features to checks


def get_file_parser(add_files: bool = True, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=add_help, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    if add_files:
        files_help_text = "The files to check. If not passed any files, inspects the entire repo."
        parser.add_argument("files", nargs="*", help=files_help_text)
    return parser


def enable_incremental(
    *match_patterns: str, exclude: Optional[Union[str, Iterable[str]]] = None
) -> Callable[[Callable[..., int]], Callable[..., int]]:
    """
    Decorator enabling an incremental version of a check.

    If a script is normally called by check/[script].py, this decorator allows it to be run with the
    arguments [-i|--incremental rev1 rev2 ...] to run the script on the changes between HEAD and the
    most recent common ancestor of rev1 rev2 ...

    Excludes integration tests by default.
    """

    def incremental_decorator(func: Callable[..., int]) -> Callable[..., int]:
        """Inner decorator that uses match_patterns."""

        def incremental_func(
            *args: Any,
            files: Optional[Iterable[str]] = None,
            revisions: Optional[Iterable[str]] = None,
            **kwargs: Any,
        ) -> int:
            silent = revisions is not None  # if passed revisions explicitly, run in silent mode

            _help = (
                "Run an incremental check on files that have changed since a specified revision.  "
                + f"If no revisions are specified, compare against the first of {default_branches} "
                + "that exists.  If multiple revisions are provided, this script compares against "
                + "their most recent common ancestor.  Incremental checks ignore integration tests."
            )

            def _add_incremental_arg(parser: argparse.ArgumentParser) -> None:
                parser.add_argument(
                    "-i", "--incremental", dest="revisions", nargs="*", action="extend", help=_help
                )

            # add incremental flags to the parser of func (so that they appear in the help text)
            func_has_parser = "parser" in inspect.signature(func).parameters
            if func_has_parser:
                parser = kwargs.get("parser", get_file_parser())
                _add_incremental_arg(parser)
                kwargs["parser"] = parser

            if revisions is None:
                # parse arguments to identify revisions to compare against
                inc_parser = argparse.ArgumentParser(add_help=not func_has_parser)
                _add_incremental_arg(inc_parser)
                inc_parsed_args, unknown_args = inc_parser.parse_known_intermixed_args(args)
                args = tuple(unknown_args)
                revisions = inc_parsed_args.revisions

            if revisions is not None:
                # add files that have changed since the most recent common ancestor of the revisions
                changed_files = get_changed_files(
                    match_patterns, revisions, silent=silent, exclude=exclude
                )
                if changed_files:
                    files = list(files) + changed_files if files else changed_files

            return func(*args, files=files, **kwargs)

        return incremental_func

    return incremental_decorator


def enable_exit_on_failure(func: Callable[..., int]) -> Callable[..., int]:
    """
    Decorator optionally allowing a function to exit instead of returning a failing return code.
    """

    def func_with_exit(*args: Any, exit_on_failure: bool = False, **kwargs: Any) -> int:
        returncode = func(*args, **kwargs)
        if exit_on_failure and returncode:
            exit(returncode)
        return returncode

    return func_with_exit


def extract_file_args(func: Callable[..., int]) -> Callable[..., int]:
    """Decorator to extract files from the arguments to a function."""

    def func_with_files(*args: Any, files: Optional[Iterable[str]] = None, **kwargs: Any) -> int:
        file_args = []
        othr_args = []
        for arg in args:
            if os.path.isfile(arg):
                file_args.append(os.path.relpath(arg, root_dir))
            else:
                othr_args.append(arg)

        if file_args:
            files = list(files) + file_args if files else file_args
        return func(*othr_args, files=files, **kwargs)

    return func_with_files
