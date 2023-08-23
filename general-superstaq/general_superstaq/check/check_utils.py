"""Dumping ground for check script utilities."""
import argparse
import enum
import fnmatch
import os
import re
import subprocess
import sys
from typing import Any, Callable, Iterable, List, Union

# identify the root directory of the "main" script that called this module
try:
    main_file_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    root_dir = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=main_file_dir, text=True
    ).strip()
except subprocess.CalledProcessError:
    root_dir = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()


def _check_output(*commands: str) -> str:
    """Wrapper for subprocess.check_output to run commands from root_dir and clean up the output."""
    return subprocess.check_output(commands, text=True, cwd=root_dir).strip()


# container for string formatting console codes
class Style(str, enum.Enum):
    """Container for string formatting console codes."""

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
    """Style (format) text for printing in the console.

    Args:
        text: The text to style.
        style_code: Console code for the text style.

    Returns:
        Styled text.
    """
    return style_code + text + Style.RESET


def warning(text: str) -> str:
    """Style (format) text indicating a warning.

    Args:
        text: The text to style.

    Returns:
        Styled text.
    """
    return styled(text, Style.BOLD + Style.YELLOW)


def failure(text: str) -> str:
    """Style (format) text indicating a failure.

    Args:
        text: The text to style.

    Returns:
        Styled text.
    """
    return styled(text, Style.BOLD + Style.RED)


def success(text: str) -> str:
    """Style (format) text indicating success.

    Args:
        text: The text to style.

    Returns:
        Styled text.
    """
    return styled(text, Style.BOLD + Style.GREEN)


# default branches to compare against when performing incremental checks
default_branches = ("upstream/main", "origin/main", "main")


####################################################################################################
# methods for identifying files to check


def get_tracked_files(include: Union[str, Iterable[str]]) -> List[str]:
    """Identify all files tracked by git (in this repo) that match the given match_patterns.

    If no patterns are provided, return a list of all tracked files in the repo.

    Args:
        include: The string of patterns to match files to.

    Returns:
        Output of `subprocess.check_output` wrapper.
    """
    include = [include] if isinstance(include, str) else include
    return _check_output("git", "ls-files", "--deduplicate", *include).splitlines()


def existing_files(files: Iterable[str]) -> List[str]:
    """Returns the subset of `files` which actually exist.

    Args:
        files: The files to check existence of.

    Returns:
        A list of existing files.
    """
    return [file for file in files if os.path.isfile(os.path.join(root_dir, file))]


def exclude_files(files: Iterable[str], exclude: Union[str, Iterable[str]]) -> List[str]:
    """Returns the files which don't match any of the globs in exclude.

    Args:
        files: The files to check for exclusion.
        exclude: The files to exclude.

    Returns:
        The excluded files.
    """
    exclude = [exclude] if isinstance(exclude, str) else exclude

    files = list(files)
    for exclusion in exclude:
        files = [file for file in files if not fnmatch.fnmatch(file, exclusion)]

    return files


def select_files(files: Iterable[str], include: Union[str, Iterable[str]]) -> List[str]:
    """Returns the files which match at least one of the globs in include.

    Args:
        files: The files to check for matching with inclusion pattern.
        include: The patterns to match for selecting input files.

    Returns:
        The selected files.
    """

    files = list(files)
    excluded_files = exclude_files(files, include)
    return [file for file in files if file not in excluded_files]


def get_changed_files(
    files: Iterable[str], revisions: Iterable[str], silent: bool = False
) -> List[str]:
    """Returns the files that have been changed in the current branch.

    You can specify git revisions to compare against when determining whether a file is considered
    to have "changed".  If multiple revisions are provided, this script compares against their most
    recent common ancestor.

    If an empty list of revisions is specified, this script will default to the first of the
    default_branches (specified above) that it finds.  If none of these branches exists, this method
    raises a ValueError.

    Args:
        files: The files to check for changes in.
        revisions: The git revisions to compare against for file checking.
        silent: An indicator for whether to print additional info about common ancestors.

    Returns:
        A list of files that were changed in the current branch.

    Raises:
        ValueError: If there are any invalid revision arguments.
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

    files_to_examine = [file for file in files if file in changed_files]

    if not silent:
        print(f"Found {len(files_to_examine)} changed file(s) to examine")
        for file in files_to_examine:
            print(file)
    return files_to_examine


def _get_ancestor(*revisions: str, silent: bool = False) -> str:
    """Helper function to identify the most recent common ancestor of the given git revisions."""
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


def get_test_files(
    files: Iterable[str], exclude: Union[str, Iterable[str]] = (), silent: bool = False
) -> List[str]:
    """For the given files, identify all associated test files.

    I.e. test files are those with the same name, but with a "_test.py" suffix).

    Args:
        files: The list of files to search for associated test files of.
        exclude: The files to exclude in the search.
        silent: An indicator of whether or not to add additional prints.

    Returns:
        A list of test files corresponding to the input files.
    """

    test_files = []
    for file in files:
        if file.split("::")[0].endswith("_test.py"):
            test_files.append(file)
        else:
            test_file = re.sub(r"\.py$", "_test.py", file)
            if os.path.isfile(os.path.join(root_dir, test_file)):
                test_files.append(test_file)
            elif not silent:
                print(warning(f"WARNING: no test file found for {file}"))

    if exclude:
        test_files = exclude_files(test_files, exclude)

    return sorted(set(test_files))


####################################################################################################
# file parsing, incremental checks, and decorator to exit instead of returning a failing exit code

CHECK_LIST = [
    "configs",
    "format",
    "flake8",
    "pylint",
    "mypy",
    "pytest",
    "coverage",
    "requirements",
    "build_docs",
]


def get_check_parser(no_files: bool = False) -> argparse.ArgumentParser:
    """Construct a command-line argument parser common to all check scripts.

    This parser collects a list of files to check, with "no files" == "all files".

    In addition, this parser has flags/arguments to:
    - skip certain (specified) checks.
    - run incremental checks, i.e., on the files that have changed since a specified git revision,
    - exclude files matching a specified glob.

    Args:
        no_files: Ignore file-related arguments.

    Returns:
        A console argument parser.
    """
    parser = argparse.ArgumentParser(
        allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--skip",
        choices=CHECK_LIST,
        nargs="+",
        action="extend",
        default=[],
        help="The checks to skip.",
    )

    if no_files:
        return parser

    help_text = "The files to check. If not passed any files, inspects the entire repo."
    parser.add_argument("files", nargs="*", help=help_text)

    help_text = (
        "Run an incremental check on files that have changed since a specified revision.  "
        f"If no revisions are specified, compare against the first of {default_branches} "
        "that exists. If multiple revisions are provided, this script compares against "
        "their most recent common ancestor. Incremental checks ignore integration tests."
    )
    parser.add_argument(
        "-i",
        "--incremental",
        dest="revisions",
        action="extend",
        nargs="*",
        help=help_text,
    )

    parser.add_argument(
        "-x",
        "--exclude",
        action="extend",
        nargs="+",
        metavar="GLOB",
        help="Exclude files matching GLOB.",
    )

    return parser


def extract_files(
    parsed_args: argparse.Namespace,
    include: Union[str, Iterable[str]] = (),
    exclude: Union[str, Iterable[str]] = (),
    silent: bool = False,
) -> List[str]:
    """Collect a list of files to test, according to command line arguments and `include`/`exclude`
    values.

    Args:
        parsed_args: The namespace generated by the ArgumentParser returned by `get_check_parser()`.
        include: Glob(s) indicating which tracked files to consider (e.g. "*.py").
        exclude: Glob(s) indicating which tracked files to skip (e.g. "*integration_test.py").
        silent: If True, restrict printing to warning and error messages.

    Returns:
        If `parsed_args.files` is empty (i.e. no file paths or globs have been passed to the file
        parser), a list of tracked files in the active repo branch meeting all of the following
        criteria:
        1. The file's path (relative to `root_dir`) matches at least one path or glob in `include`,
        2. The path does not match any path or glob in `exclude`,
        3. The path does not match any path or glob in `parsed_args.exclude`,
        4. If `parsed_args.revisions` is not None, the file additionally must have been modified in
            the current branch (see `get_changed_files()` for details on how this is determined).

        If `parsed_args.files` is nonempty, a restricted file list containing:
        1. Paths meeting the above criteria and matching any glob in `parsed_args.files`.
        2. Paths meeting the above criteria and located in any extant subdirectory passed to the
            file parser directly.
        3. Paths to extant files passed to the file parser directly, regardless of above criteria.
    """

    exclude = [exclude] if isinstance(exclude, str) else exclude

    if parsed_args.exclude:
        exclude = [*exclude, *parsed_args.exclude]

    files = []
    globs = []
    for glob in parsed_args.files:
        if os.path.isfile(glob.split("::")[0]):  # always include files passed directly as arguments
            files.append(os.path.relpath(glob, start=root_dir))
        elif os.path.isdir(glob):  # treat subdirectories the same as the glob "<subdir>/*"
            globs.append(os.path.normpath(os.path.join(os.path.relpath(glob, start=root_dir), "*")))
        else:
            globs.append(glob)

    if globs or not parsed_args.files:
        tracked_files = get_tracked_files(include)
        tracked_files = existing_files(tracked_files)
        tracked_files = exclude_files(tracked_files, exclude)
        if globs:
            tracked_files = select_files(tracked_files, globs)
        files += tracked_files

    if parsed_args.revisions is not None:
        files = get_changed_files(files, parsed_args.revisions, silent=silent)

    return sorted(set(files))


def enable_exit_on_failure(func_with_returncode: Callable[..., int]) -> Callable[..., int]:
    """Decorator optionally allowing a function to exit instead of failing.

    I.e., catches a failing return code for safely exiting function.

    Args:
        func_with_returncode: A function with a return code for failure.

    Returns:
        A decorated function that handles failing return codes.
    """

    def func_with_exit(*args: Any, exit_on_failure: bool = False, **kwargs: Any) -> int:
        """A function that exits safely on failure.

        Args:
            args: Any arguments for the function to be decorated.
            exit_on_failure: An indicator of whether the function will exit on failure.
            kwargs: Any keyword arguments for the function to be decorated..

        Returns:
            The return code.
        """
        returncode = func_with_returncode(*args, **kwargs)
        if exit_on_failure and returncode:
            exit(returncode)
        return returncode

    return func_with_exit
