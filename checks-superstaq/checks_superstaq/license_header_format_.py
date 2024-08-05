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

import dataclasses
import enum
import re
import sys
import textwrap
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import tomlkit

from checks_superstaq import check_utils


def read_toml() -> tuple[str, str, str, bool] | None:
    """Reads the pyproject.toml file to get license information. Fields should be under
    [tool.license_header_format] and named `license_header`, `license_name`, `licensee` and
    `editable`.

    Returns:
        Tuple containing the exptected license header, license name, licensee, and
    whether similar licenses are editable.
    """

    data: dict[str, Any] = tomlkit.parse(Path("pyproject.toml").read_text())
    try:
        expected_license_header = str(data["tool"]["license_header_format"]["license_header"])
    except KeyError:
        print(
            "Under [tool.license_header_format] add a `license_header` field filled with the"
            "license header that should be added to source code files in the repository."
        )
        return None

    try:
        license_name = str(data["tool"]["license_header_format"]["license_name"])
    except KeyError:
        print(
            "Under [tool.license_header_format] add a `license_name` field filled with the"
            " license's name."
        )
        return None

    try:
        licensee = str(data["tool"]["license_header_format"]["licensee"])
    except KeyError:
        print(
            "Under [tool.license_header_format] add a `licensee` field filled with the "
            " name of the licensee."
        )
        return None

    try:
        editable = bool(data["tool"]["license_header_format"]["editable"])
    except KeyError:
        print(
            "Under [tool.license_header_format] add an `editable` boolean field set to True if the"
            " license owner can be appended to similar license and False otherwise."
        )
        return None

    return expected_license_header, license_name, licensee, editable


class HeaderType(enum.Enum):
    """Enum used to store the types of licence headers that be found in source code files.

    - VALID: valid license header.
    - SIMILAR_LICENSE: Matching license type but different licensee.
    - OUTDATED: A license belonging to the same licensee but different type than the target.
    - OTHER: Any other licenses.
    """

    VALID = 1
    SIMILAR_LICENSE = 2
    OUTDATED = 3
    OTHER = 4


@dataclasses.dataclass
class LicenseHeader:
    """Class to describe license headers found in files including the header itself, the line
    numbers where it is found in a file, and the type of the license header."""

    start_line_num: int
    header_type: HeaderType | None = None
    end_line_num: int = 0
    license_header: str = ""

    def __str__(self) -> str:
        """The string representation of a license header used later for printing."""
        return (
            f"Beginning at line: {self.start_line_num}\n"
            f"Ending at line   : {self.end_line_num}\n\n"
            f"{self.license_header}\n"
        )


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

    with open(file, "r") as f:
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


def _validate_license_header(
    license_header_lst: list[LicenseHeader],
    expected_license_header: str,
    licensee: str,
    license_name: str,
    editable: bool,
) -> bool:
    """Returns whether there is a valid license header in a file and for each license
    header in a file, it assigns each theiir type.
        - VALID: if the header contains a Copyright line.
        - OUTDATED: if the header has the correct licensee but different type.
        - SIMILAR_LICENSE: if the header is a valid license but not from the licensee.
        - OTHER: if the header is any other one.
    Args:
        license_header_lst: List of the license_header objects for each header in a file.
        licensee: The owner of the expected license.
        license_name: The name of the expected license.
        editable: Whether similar licenses can be edited to include the license owner instead
            adding the entire license to the file.

    Returns: Whether there is a valid license header in a file or not.
    """
    target = (
        expected_license_header.replace("{YEAR}", ".*")
        .replace("{LICENSEE}", licensee)
        .replace("\n", "")
    )
    print(target)
    valid_header_regex = re.compile(rf"{target}")
    valid = False

    for license_header in license_header_lst:
        print(re.match(valid_header_regex, license_header.license_header.replace("\n", "")))
        print(license_header.license_header.replace("\n", ""))
        if (
            re.match(valid_header_regex, license_header.license_header)
            and license_name in license_header.license_header
        ):
            license_header.header_type = HeaderType.VALID
            valid = True
        elif (
            re.search(valid_header_regex, license_header.license_header)
            and license_name not in license_header.license_header
        ):
            license_header.header_type = HeaderType.OUTDATED
        elif not editable or license_name not in license_header.license_header:
            license_header.header_type = HeaderType.OTHER
        else:
            license_header.header_type = HeaderType.SIMILAR_LICENSE

    return valid


def _append_to_header(file: str, license_header: LicenseHeader, licensee: str) -> None:
    """Appends licensee to existing license header.

    Args:
        file: The name/path for the file whose license header will have the licensee added to it.
        license_header: The specific license header that the licensee is being appended to.
        licensee: The licensee of the target license header.

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
                    prepend += line[:-1] + f" 2024 {licensee}.\n"
                elif line[-2].isalpha():
                    prepend += line[:-1] + f", 2024 {licensee}.\n"
                else:
                    prepend += line[:-2] + f", 2024 {licensee}.\n"
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


def _add_license_header(file: str, expected_license_header: str) -> None:
    """Adds the correct license header to a file.

    Args:
        file: The file name/path to which license header is added.
        expected_license_header: The target license header.

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
    licensee: str,
) -> bool:
    """Function handles bad headers in files. The cases are handled according to the HeaderType:
        - VALID and OTHER: no change.
        - SIMILAR_LICENSE: if the header is a valid license but not from the licensee.
        - OUTDATED: will be removed

    Args:
        file: The file name/path from which the bad license header is removed.
        styled_file_name: Styled file name.
        license_header: The LicenseHeader object of the currect header being handled.
        apply: Whether to fix the license header if it is incorrect.
        valid: Whether there is a valid header in the file.
        silent: Whether to print out any incorrect license headers that have been found.
        append_flag: Whether the licensee has already been appended to a header in the file.
        licensee: The owner of the expected license.

    Returns:
        The updated append_flag.
    """
    if license_header.header_type == HeaderType.SIMILAR_LICENSE:
        if not valid and not silent and not apply:
            print("----------")
            print(check_utils.warning(str(license_header)))
            print("----------")
        # don't append licensee to license header if there is a valid license header already or
        # it has already been appended to another license header.
        if not append_flag and apply and not valid:
            _append_to_header(file, license_header, licensee)
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


def run_checker(
    file: str,
    apply: bool,
    silent: bool,
    no_header: bool,
    bad_header: bool,
    expected_license_header: str,
    license_name: str,
    licensee: str,
    editable: bool,
) -> int:
    """For a given file, checks if it has the correct license header. If apply is set to True,
    it removes any bad license headers that have been found and replaces them with the correct
    license header.

    Args:
        file: The file name/path from which the bad license header is removed.
        apply: Whether to fix the license header if it is incorrect.
        silent: Whether to print out any incorrect license headers that have been found.
        no_header: Whether to only handle files with no license headers.
        bad_header: Whether to only handle files with incorrect headers.
        expected_license_header: The target license header.
        license_name: The name of the expected license.
        licensee: The owner of the expected license.
        editable: Whether similar licenses can be edited to include the license owner instead
            adding the entire license to the file.

    Returns:
        The exit code. 1 if an incorrect or no license header is found. 0 if correct.
    """

    license_header_lst: list[LicenseHeader] = _extract_license_header(file)
    styled_file_name = check_utils.styled(file, check_utils.Style.BOLD)

    if len(license_header_lst) == 0:
        if (not no_header and not bad_header) or no_header:
            if apply:
                _add_license_header(file, expected_license_header)
                print(f"{styled_file_name}: {check_utils.success('License header added.')}")
            else:
                print(f"{styled_file_name}: {check_utils.warning('No license header found.')}")
            return 1
        else:
            return 0

    if no_header and not bad_header:  # if the --no-header flag is set
        return 0

    valid = _validate_license_header(
        license_header_lst, expected_license_header, licensee, license_name, editable
    )
    append_flag = False  # ensure the licensee is not appended to multiple headers
    exit_code = 0

    # Incorrect license header if it has no valid license header or has an outdated one.
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
            licensee,
        )

    if not valid and not append_flag and apply:
        _add_license_header(file, expected_license_header)
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
        "--apply",
        action="store_true",
        help="Add the license header to files.",
    )

    target_case = parser.add_mutually_exclusive_group()
    target_case.add_argument(
        "--no-header",
        action="store_true",
        help="Hanlde only files with no license header.",
    )
    target_case.add_argument(
        "--bad-header",
        action="store_true",
        help="Handle only files with incorrect license headers.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Do not show incorrect license headers.",
    )

    parsed_args, _ = parser.parse_known_intermixed_args(args)
    if "license_check" in parsed_args.skip:
        return 0

    files = check_utils.extract_files(parsed_args, include, exclude, silent)
    if not files:
        print("No files selected.\n")
        return 0

    if (toml_data := read_toml()) is not None:
        expected_license_header, licensee, license_name, editable = toml_data
    else:
        return 0

    exit_code = 0
    for file in files:
        exit_code += run_checker(
            file,
            parsed_args.apply,
            parsed_args.silent or silent,
            parsed_args.no_header,
            parsed_args.bad_header,
            expected_license_header,
            licensee,
            license_name,
            editable,
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
