#!/usr/bin/env python3
import argparse
import difflib
import os
import sys
import textwrap
from typing import List, Tuple

from general_superstaq.check import check_utils


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    config_file: str = "setup.cfg",
    ignore_match: str = "# REPO-SPECIFIC CONFIG",
    silent: bool = False,
) -> int:

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.description = textwrap.dedent(
        f"""
        Checks that the check script configuration file ({config_file}) is consistent across repos.

        Exceptions in {config_file} can be flagged by adding "{ignore_match}" to the line.
        """
    )
    parser.parse_args(args)  # placeholder parsing to enable printing help text

    # identify the "original" config file, and the file that is supposed to be a copy
    file_orig = os.path.join(os.path.abspath(os.path.dirname(__file__)), config_file)
    file_copy = os.path.join(check_utils.root_dir, config_file)
    lines_orig = open(file_orig, "r").read().splitlines()
    lines_copy = open(file_copy, "r").read().splitlines()

    # collect differences between config files, ignoring lines in file_copy that match ignore_match
    matcher = difflib.SequenceMatcher(a=lines_orig, b=lines_copy)
    deltas: List[Tuple[str, int, int, int, int]] = []
    for tag, orig_start, orig_end, copy_start, copy_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        diff_lines_copy = lines_copy[copy_start:copy_end]
        if not diff_lines_copy or any(ignore_match not in line for line in diff_lines_copy):
            deltas.append((tag, orig_start, orig_end, copy_start, copy_end))

    if not deltas:
        if not silent:
            print(check_utils.success("Config files are consistent!"))
        return 0

    # print diffs
    print(check_utils.failure(f"Found {len(deltas)} differences between config files:"))
    print(check_utils.styled(f"< {file_orig} (original)", check_utils.Style.RED))
    print(check_utils.styled(f"> {file_copy} (copy)", check_utils.Style.GREEN))
    print(check_utils.styled("-" * 70, check_utils.Style.CYAN))
    for tag, orig_start, orig_end, copy_start, copy_end in deltas:
        _announce_diff(tag, orig_start, orig_end, copy_start, copy_end)
        if orig_start != orig_end:
            text = "\n".join(f"< {line}" for line in lines_orig[orig_start:orig_end])
            print(check_utils.styled(text, check_utils.Style.RED))
        if copy_start != copy_end:
            text = "\n".join(f"> {line}" for line in lines_copy[copy_start:copy_end])
            print(check_utils.styled(text, check_utils.Style.GREEN))
    print(check_utils.failure(f"{os.path.relpath(file_copy)} must be fixed manually!"))
    return 1


def _announce_diff(
    tag: str, orig_start: int, orig_end: int, copy_start: int, copy_end: int
) -> None:
    assert tag in ["replace", "delete", "insert"]
    line_text_orig = _line_text(orig_start, orig_end)
    line_text_copy = _line_text(copy_start, copy_end)
    if tag == "replace":
        text = f"{line_text_orig} of original replaced by {line_text_copy} of copy"
    elif tag == "delete":
        text = f"{line_text_orig} of original not present in copy"
    elif tag == "insert":
        text = f"{line_text_copy} of copy not present in original"
    print(check_utils.styled(text, check_utils.Style.CYAN))


def _line_text(start: int, end: int) -> str:
    if start + 1 == end:
        return f"line {start+1}"
    return f"lines {start+1}--{end}"


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
