#!/usr/bin/env python3

import subprocess
import sys
import textwrap
from typing import Iterable, Union

from general_superstaq.check import check_utils


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: Union[str, Iterable[str]] = ("*.py", "*.ipynb"),
    exclude: Union[str, Iterable[str]] = "",
    silent: bool = False,
) -> int:

    parser = check_utils.get_file_parser()
    parser.description = textwrap.dedent(
        """
        Runs black and isort on the repository (formatting check).
        """
    )

    parser.add_argument("--apply", action="store_true", help="Apply changes to files.")

    parsed_args, args_to_pass_black = parser.parse_known_intermixed_args(args)
    files = check_utils.extract_files(parsed_args, include, exclude, silent)

    diff_check_args = ["--diff", "--check"] if not parsed_args.apply else []
    returncode_black = subprocess.call(
        ["black", *files, *diff_check_args, *args_to_pass_black], cwd=check_utils.root_dir
    )

    if returncode_black > 1:
        # this only occurs if black could not parse a file (for example due to a syntax error)
        return returncode_black

    returncode_isort = subprocess.call(
        ["isort", *files, *diff_check_args], cwd=check_utils.root_dir
    )

    if returncode_black == 1 or returncode_isort == 1:
        # some files should be reformatted, but there don't seem to be any bona fide errors
        command = "./check/format_.py --apply"
        text = f"Run '{command}' (from the repo root directory) to format files."
        print(check_utils.warning(text))
        return 1

    return returncode_isort


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
