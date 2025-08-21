#!/usr/bin/env python3
# Copyright 2025 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from checks_superstaq import check_utils


@check_utils.enable_exit_on_failure
def run(*args: str) -> int:
    """Checks that the docs build successfully.

    Args:
        *args: Command line arguments.

    Returns:
        Terminal exit code. 0 indicates success, while any other integer indicates a test failure.
    """
    parser = check_utils.get_check_parser(no_files=True)
    parser.description = textwrap.dedent(
        """
        Checks that the docs build successfully.
        """
    )
    parsed_args, _ = parser.parse_known_intermixed_args(args)
    if "build_docs" in parsed_args.skip:
        return 0

    docs_dir = os.path.join(check_utils.root_dir, "docs")
    if not os.path.isdir(os.path.join(docs_dir, "source")):
        print(check_utils.warning("No docs to build."))  # noqa: T201
        return 0
    return subprocess.call(
        ["sphinx-build", "source", "build/html", "--fail-on-warning", "--keep-going"], cwd=docs_dir
    )


if __name__ == "__main__":
    sys.exit(run(*sys.argv[1:]))
