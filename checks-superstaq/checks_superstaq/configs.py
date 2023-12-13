#!/usr/bin/env python3
import difflib
import os
import sys
import textwrap
from typing import List, Tuple

from checks_superstaq import check_utils

CONFIG_FILE = "pyproject.toml"
TEMPLATE_FILE = "checks-pyproject.toml"
START_MATCH = "# Check script configuration:"
IGNORE_MATCH = "# REPO-SPECIFIC CONFIG"


@check_utils.enable_exit_on_failure
def run(*args: str, silent: bool = False) -> int:
    """Checks that check-script configurations are consistent across repos.

    Args:
        *args: Command line arguments.
        silent: If True, restrict printing to warning and error messages.

    Returns:
        Terminal exit code. 0 indicates success, while any other integer indicates a test failure.
    """

    parser = check_utils.get_check_parser(no_files=True)
    parser.description = textwrap.dedent(
        f"""
        Checks that the check script configuration file ({CONFIG_FILE}) is consistent across repos.

        Exceptions in {CONFIG_FILE} can be flagged by adding "{IGNORE_MATCH}" to the line.
        """
    )
    parsed_args, _ = parser.parse_known_intermixed_args(args)
    if "configs" in parsed_args.skip:
        return 0

    # identify the "original" config file, and the file that is supposed to be a copy
    file_orig = os.path.join(os.path.abspath(os.path.dirname(__file__)), TEMPLATE_FILE)
    file_copy = os.path.join(check_utils.root_dir, CONFIG_FILE)
    lines_orig = open(file_orig, "r").read().splitlines()
    lines_copy = open(file_copy, "r").read().splitlines()

    # trim package-specific configuration
    lines_orig, orig_offset = _trim_lines(lines_orig)
    lines_copy, copy_offset = _trim_lines(lines_copy)

    # collect differences between config files, ignoring lines in file_copy that match IGNORE_MATCH
    matcher = difflib.SequenceMatcher(a=lines_orig, b=lines_copy)
    deltas: List[Tuple[str, int, int, int, int]] = []
    for tag, orig_start, orig_end, copy_start, copy_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        diff_lines_copy = lines_copy[copy_start:copy_end]
        if not diff_lines_copy or any(IGNORE_MATCH not in line for line in diff_lines_copy):
            deltas.append((tag, orig_start, orig_end, copy_start, copy_end))

    if not deltas:
        if not silent:
            print(check_utils.success("Config files are consistent!"))
        return 0

    # print diffs
    num_differences = f"{len(deltas)} differences" if len(deltas) > 1 else "one difference"
    print(check_utils.warning(f"WARNING: found {num_differences} between config files:"))
    print(check_utils.styled(f"< {file_orig} (original)", check_utils.Style.RED))
    print(check_utils.styled(f"> {file_copy} (copy)", check_utils.Style.GREEN))
    print(check_utils.styled("-" * 70, check_utils.Style.CYAN))
    for tag, orig_start, orig_end, copy_start, copy_end in deltas:
        _announce_diff(tag, orig_offset, orig_start, orig_end, copy_offset, copy_start, copy_end)
        if orig_start != orig_end:
            text = "\n".join(f"< {line}" for line in lines_orig[orig_start:orig_end])
            print(check_utils.styled(text, check_utils.Style.RED))
        if copy_start != copy_end:
            text = "\n".join(f"> {line}" for line in lines_copy[copy_start:copy_end])
            print(check_utils.styled(text, check_utils.Style.GREEN))

    file_path = os.path.relpath(file_copy)
    print(check_utils.warning(f"{file_path} can be fixed manually to prevent this message."))
    return 0


def _trim_lines(lines: List[str]) -> Tuple[List[str], int]:
    """Remove package-specific configuration text, and identify the starting lines to compare."""
    for start_line, line in enumerate(lines):
        if START_MATCH in line:
            return lines[start_line:], start_line
    return lines, 0


def _announce_diff(
    tag: str,
    orig_offset: int,
    orig_start: int,
    orig_end: int,
    copy_offset: int,
    copy_start: int,
    copy_end: int,
) -> None:
    """Announce a difference between configuration files."""
    assert tag in ["replace", "delete", "insert"]
    line_text_orig = _line_text(orig_start + orig_offset, orig_end + orig_offset)
    line_text_copy = _line_text(copy_start + copy_offset, copy_end + copy_offset)
    if tag == "replace":
        text = f"{line_text_orig} of original replaced by {line_text_copy} of copy"
    elif tag == "delete":
        text = f"{line_text_orig} of original not present in copy"
    elif tag == "insert":
        text = f"{line_text_copy} of copy not present in original"
    print(check_utils.styled(text, check_utils.Style.CYAN))


def _line_text(start: int, end: int) -> str:
    """Identify one or multiple line numbers."""
    if end == start + 1:
        return f"line {start+1}"
    return f"lines {start+1}--{end}"


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
