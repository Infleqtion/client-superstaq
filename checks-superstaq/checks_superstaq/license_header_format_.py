#!/usr/bin/env python3

# Copyright 2024 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import enum
import re
import sys
import textwrap
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import tomlkit

from checks_superstaq import check_utils

# The license header that should be added to the files with no license headers is read from
# the pyproject.toml file. It should be under [tool.license_header_format] assigned to the variable
# license_header
try:
    data: dict[str, Any] = tomlkit.parse(Path("pyproject.toml").read_text())
    expected_license_header = str(data["tool"]["license_header_format"]["license_header"])
    in_server = "Apache" not in expected_license_header
except KeyError:
    raise KeyError(
        "Under [tool.license_header_format] add a license_header field with the license\
 heder that should be added to source code files in the repository."
    )


class HeaderType(enum.Enum):
    """Enum used to store the types of licence headers that be found in source code files.

    - VALID: valid Infleqtion license header.
    - OTHER_APACHE: An Apache license header that is not Infleqtion's.
    - OUTDATED: A license belonging to ColdQuanta Inc.
    - OTHER: Any other licenses.
    """

    VALID = 1
    OTHER_APACHE = 2
    OUTDATED = 3
    OTHER = 4


class LicenseHeader:
    """Class to describe license headers found in files including the header itself, the line
    numbers where it is found in a file, and the type of the license header."""

    def __init__(self, start_line_num: int) -> None:
        self.start_line_num = start_line_num
        self.license_header = ""
        self.end_line_num = 0

    @property
    def header_type(self) -> HeaderType:
        """Returns the type of the license header."""
        return self._header_type

    @header_type.setter
    def header_type(self, header_type: HeaderType) -> None:
        """Sets the type of the license header."""
        self._header_type = header_type

    def __str__(self) -> str:
        """The string representation of a license header used later for printing."""
        return f"""
    Beginning at line: {self.start_line_num}
    Ending at line   : {self.end_line_num}\n
{self.license_header}\n"""


def _extract_license_header(file: str) -> list[LicenseHeader]:
    """Extracts the license headers from a file. Reads the file until it finds a none comment line.
    Pylint and mypy disabling comments and shebangs are ignored. White spaces preciding the
    license header and between the header and shebang and/or pylint line are also ignored.
    Also checks if the comment block selected contains the keyword 'Copyright'.

    Args:
        file: The file name/path.

    Returns a list of LicenseHeader object each for being the distinct license headers found in
    the file.
    """
    license_header_lst: list[LicenseHeader] = []
    license_header = ""
    exceptions = ["# pylint:", "#!/", "# mypy:"]

    with open(file, "r+") as f:
        for line_num, line in enumerate(f):
            if not license_header and line[0] == "\n":
                continue
            if line[0] != "#" and line[0] != "\n":
                if license_header:
                    license_header_lst[-1].license_header = license_header
                    license_header_lst[-1].end_line_num = line_num + 1
                    license_header = ""
                break

            if all(exception not in line for exception in exceptions):
                if not license_header:
                    license_header_lst.append(LicenseHeader(line_num + 1))

                if line == "\n":
                    # set the line number for the last line of the license_header
                    license_header_lst[-1].license_header = license_header
                    license_header_lst[-1].end_line_num = line_num + 1
                    license_header = ""
                else:
                    license_header += line
    license_header_lst = [
        header for header in license_header_lst if "Copyright" in header.license_header
    ]

    return license_header_lst


def _validate_license_header(license_header_lst: list[LicenseHeader]) -> bool:
    """Returns whether there is a valid Infleqtion license header in a file and for each license
    header in a file, it assigns each theiir type.
        - VALID: if the header contains a Copyright Infleqtion line.
        - OUTDATED: if the header is for ColdQuanta Inc.
        - OTHER_APACHE: if the header is an Apache license but not from Infleqtion
                for client-superstaq.
        - OTHER: if the header is any other one. Also includes Apache license headers for
                server-superstaq.
    Args:
        license_header_lst: List of the license_header objects for each header in a file.

    Returns: Whether there is a valid license header in a file or not.
    """
    valid_header_regex = re.compile(r"(.*)Copyright(.*)Infleqtion")
    outdated_header_regex = re.compile(r"(.*)Copyright(.*)ColdQuanta Inc\.")
    valid = False

    for license_header in license_header_lst:
        if re.search(outdated_header_regex, license_header.license_header):
            license_header.header_type = HeaderType.OUTDATED
        elif re.search(valid_header_regex, license_header.license_header):
            license_header.header_type = HeaderType.VALID
            valid = True
        elif in_server or "Apache" not in license_header.license_header:
            license_header.header_type = HeaderType.OTHER
        else:
            license_header.header_type = HeaderType.OTHER_APACHE

    return valid


def _append_to_header(file: str, license_header: LicenseHeader) -> None:
    """Appends Infleqtion to existing Apache license that is not from Infleqtion.

    Args:
        file: The name/path for the file whose license header will have Infleqtion added to it.
        license_header: The specific license header that Infleqtion is being appended to.

    Returns nothing.
    """
    prepend = ""
    char_count = 0
    with open(file, "r+") as f:
        for line_num, line in enumerate(f):
            char_count += len(line)
            if (
                "Copyright" in line
                and license_header.start_line_num <= line_num + 1 < license_header.end_line_num
            ):
                if line[-2] == ",":
                    prepend += line[:-1] + " 2024 Infleqtion.\n"
                else:
                    prepend += line[:-2] + ", 2024 Infleqtion.\n"
                break
            prepend += line

        f.seek(char_count)
        content = f.read()
        f.seek(0)
        f.write(prepend + content)
        f.truncate()


def _remove_header(file: str, license_header: LicenseHeader) -> None:
    """Removes a license header from a file.

    Args:
        file: The file name/path from which the bad license header is removed.
        license_header: The specific license header that is being removed.

    Returns nothing.
    """
    char_count = 0
    prepend = ""

    with open(file, "r+") as f:
        for line_num, line in enumerate(f):
            if line_num + 1 < license_header.start_line_num:
                prepend += line
            if line_num + 1 == license_header.end_line_num:
                break
            char_count += len(line)

        f.seek(char_count)
        append = f.read()

        f.seek(0)
        f.write(prepend + append)
        f.truncate()


def _add_license_header(file: str) -> None:
    """Adds the correct license header to a file.

    Args:
        file: The file name/path to which license header is added.

    Returns nothing.
    """
    exceptions = ["# pylint:", "#!/", "# mypy:"]
    exception_lines = ""
    char_count = 0
    with open(file, "r+") as f:
        for line in f:
            if any(line.startswith(exception) for exception in exceptions):
                exception_lines += line
                char_count += len(line)
            else:
                break

        f.seek(char_count)
        content = f.read()
        f.seek(0)
        f.write(exception_lines + expected_license_header + content)
        f.truncate()


def handle_bad_header(
    file: str,
    styled_file_name: str,
    license_header: LicenseHeader,
    apply: bool,
    valid: bool,
    silent: bool,
    append_flag: bool,
) -> bool:
    """Function handles bad headers in files. The cases are handled according to the HeaderType:
        - VALID and OTHER: no change.
        - OTHER_APACHE: the first of this type will have Infleqtion appended to license header.
        - OUTDATED: will be removed

    Args:
        file: The file name/path from which the bad license header is removed.
        styled_file_name: Styled file name.
        license_header: The LicenseHeader object of the currect header being handled.
        apply: Whether to fix the license header if it is incorrect.
        valid: Whether there is a valid header in the file.
        silent: Whether to print out any incorrect license headers that have been found.
        append_flag: Whether Infleqtion has already been appended to an Apache License in the file.

    Returns the updated append_flag.
    """
    if license_header.header_type == HeaderType.OTHER_APACHE:
        if not valid and not silent and not apply:
            print("----------")
            print(check_utils.warning(str(license_header)))
            print("----------")
        # don't append Infleqtion to Apache license if there is a valid Infleqtion
        # license header already or it has already been appended to a license.
        if not append_flag and apply and not valid:
            _append_to_header(file, license_header)
            append_flag = True
            print(f"{styled_file_name}: {check_utils.success('License header fixed.')}")
    elif license_header.header_type == HeaderType.OUTDATED:
        if not silent and not apply:
            print("----------")
            print(check_utils.warning(str(license_header)))
            print("----------")
        if apply:
            _remove_header(file, license_header)
            print(f"{styled_file_name}: {check_utils.success('License header removed.')}")
    elif license_header.header_type == HeaderType.OTHER:
        if not silent and not valid and not apply:
            print("----------")
            print(check_utils.warning(str(license_header)))
            print("----------")
    return append_flag


def run_checker(file: str, apply: bool, silent: bool, no_header: bool, bad_header: bool) -> int:
    """For a given file, checks if it has the correct license header. If apply is set to True,
    it removes any bad license headers that have been found and replaces them with the correct
    license header.

    Args:
        file: The file name/path from which the bad license header is removed.
        apply: Whether to fix the license header if it is incorrect.
        silent: Whether to print out any incorrect license headers that have been found.
        no_header: Whether to only handle files with no license headers.
        bad_header: Whether to only handle files with incorrect headers.

    Returns the exit code. 1 if an incorrect or no license header is found. 0 if correct.
    """
    license_header_lst: list[LicenseHeader] = _extract_license_header(file)
    styled_file_name = check_utils.styled(file, check_utils.Style.BOLD)

    if len(license_header_lst) == 0:
        if (not no_header and not bad_header) or no_header:
            if apply:
                _add_license_header(file)
                print(f"{styled_file_name}: {check_utils.success('License header added.')}")
            else:
                print(f"{styled_file_name}: {check_utils.warning('No license header found.')}")
            return 1
        else:
            return 0
        # return handle_no_header(file, apply, no_header, bad_header)

    if no_header and not bad_header:  # if the --no-header flag is set
        return 0

    valid = _validate_license_header(license_header_lst)
    append_flag = False  # used to make sure Infleqtion is not appended to multiple Apace headers
    exit_code = 0

    # A file has an incorrect license header if it has no valid Infleqtion license header or
    # has an outdated ColdQuanta Inc license.
    if not valid or any(header.header_type == HeaderType.OUTDATED for header in license_header_lst):
        exit_code = 1
        if not apply:
            print(f"{styled_file_name}: {check_utils.warning('Incorrect license header found.')}")

    for license_header in license_header_lst:
        append_flag = handle_bad_header(
            file,
            styled_file_name,
            license_header,
            apply,
            valid,
            silent,
            append_flag,
        )

    if not valid and not append_flag and apply:
        _add_license_header(file)
        print(f"{styled_file_name}: {check_utils.success('License header added.')}")

    return exit_code


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: str | Iterable[str] = "*.py",
    exclude: str | Iterable[str] = (),
    silent: bool = False,
) -> int:
    """Sets up command line arguments and runs the license header check on all entered files.

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
        Runs the license header formatter on the repository.
        """
    )
    parser.add_argument(
        "--apply", action="store_true", help="Add the license header to files.", default=False
    )

    target_case = parser.add_mutually_exclusive_group()
    target_case.add_argument(
        "--no-header",
        action="store_true",
        help="Hanlde only files with no license header.",
        default=False,
    )
    target_case.add_argument(
        "--bad-header",
        action="store_true",
        help="Handle only files with incorrect license headers.",
        default=False,
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Do not show incorrect license headers.",
        default=False,
    )

    parsed_args, _ = parser.parse_known_intermixed_args(args)
    if "license_check" in parsed_args.skip:
        return 0

    files = check_utils.extract_files(parsed_args, include, exclude, silent)
    if not files:
        print("No files selected.\n")
        return 0

    exit_code = 0
    for file in files:
        exit_code += run_checker(
            file,
            parsed_args.apply,
            parsed_args.silent or silent,
            parsed_args.no_header,
            parsed_args.bad_header,
        )

    if not exit_code:
        print(check_utils.success("All license headers are correct!"))
    elif not parsed_args.apply:
        print(
            check_utils.warning(
                f"{exit_code} issues found.\n"
                "Run './checks/license_header_format_.py --apply' (from the repo root directory) to"
                " fix the license headers."
            )
        )

    return exit_code


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
