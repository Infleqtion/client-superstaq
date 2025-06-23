#!/usr/bin/env python3
import subprocess
import textwrap
from collections.abc import Iterable
import tomlkit

from checks_superstaq import check_utils
from pathlib import Path
from typing import Any

CONFIG_FILE = "pyproject.toml"

def read_toml() -> tuple[str, str, str] | None:
    """Reads the pyproject.toml file to get license information. Fields should be under
    [tool.add_license_header] and named `license_template`, `author_name` and
    `year`.

    Returns:
        Tuple containing the exptected license template, author name, and year.
    """

    data: dict[str, Any] = tomlkit.parse(Path(CONFIG_FILE).read_text())
    try:
        license_template = str(data["tool"]["add-license-header"]["license_template"])
    except KeyError:
        print(
            "Under [tool.add-license-header] add a `license_template` field filled with the"
            "license header that should be added to source code files in the repository."
        )
        return None

    try:
        author_name = str(data["tool"]["add-license-header"]["author_name"])
    except KeyError:
        print(
            "Under [tool.add-license-header] add an `owner` field filled with the "
            " name of the licensee."
        )
        return None

    try:
        year = str(data["tool"]["add-license-header"]["create_year"])
    except KeyError:
        print(
            "Under [tool.add-license-header] add a `create_year` field "
            "to replace $create_year. Defaults to looking up creation date in git history. " 
            "If that fails, will be defaulted to the current year."
        )
        return None

    return license_template, author_name, year

def run(
    *args: str,
    include: str | Iterable[str] = "*.py",
    exclude: str | Iterable[str] = "*.ipynb",
    silent: bool = False,
) -> int:
    """Runs the 'add-license-header' tool on the repository.

    Args:
        *args: Command line arguments.
        include: Glob(s) indicating which tracked files to consider (e.g. "*.py").
        exclude: Glob(s) indicating which tracked files to skip (e.g. "*integration_test.py").
        silent: If True, restrict printing to warning and error messages.

    Returns:
        Terminal exit code. 0 indicates success, while any other integer indicates a test failure.
    """
    parser = check_utils.get_check_parser()
    parser.description = textwrap.dedent(
        """
        Runs 'add-license-header' on the repository (adds license headers to source code)
        """
    )

    parsed_args, args_to_pass = parser.parse_known_intermixed_args(args)

    if "add_license_header" in parsed_args.skip:
        return 0

    license_template, author_name, year = read_toml()
    files = check_utils.extract_files(parsed_args, include, exclude, silent)

    cmd_args = [
        *args_to_pass,
        "--unmanaged",
        "--license", 
        license_template,
        "--author-name",
        author_name,
    ]
    
    if year:
        cmd_args.append("--create-year")
        cmd_args.append(year)

    if files:
        return_code = subprocess.call(
            ["add-license-header", *cmd_args, *files],
            cwd=check_utils.root_dir,
        )
        return return_code

    return 0