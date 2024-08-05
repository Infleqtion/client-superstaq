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
"""Check for whether python file has from __future__ import annotations as the first import."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pylint.checkers import BaseChecker
from pylint.checkers.utils import only_required_for_messages

if TYPE_CHECKING:
    from astroid import nodes
    from pylint.lint import PyLinter


class ImportAnnotationsChecker(BaseChecker):
    """Checker for whether future annotations is imported in python files."""

    name = "import-future-annotations"
    msgs = {
        "W6063": (
            "Missing from __future__ import annotations",
            "missing-annotations-import",
            "Missing annotations import causes compatibility issues with older python versions.",
        ),
    }

    def visit_module(self, _: nodes.Module) -> None:
        """Sets the flag for import annotations found to False for each module."""
        self.found_import_annotations = False

    @only_required_for_messages("import-future-annotations")
    def visit_importfrom(self, node: nodes.ImportFrom) -> None:
        """Checks if 'from __future__ import annotations' is one of the imports in a module.

        Args:
            node: All the ImportFrom nodes in a module.
        """
        self.found_import_annotations = (
            self.found_import_annotations
            or node.modname == "__future__"
            and any("annotations" in name for name in node.names)
        )

    def leave_module(self, node: nodes.Module) -> None:
        """Method reports if 'from __future__ import annotations' was found in a file that does not
        start with '_'.

        Args:
            node: The current module.
        """
        if os.path.basename(node.file).startswith("_") or self.found_import_annotations:
            return
        self.add_message("missing-annotations-import", node=node)


def register(linter: PyLinter) -> None:
    """Registers plugin to be accessed by pylint.

    Args:
        linter: The base pylinter which the custom checker will inherit from.
    """
    linter.register_checker(ImportAnnotationsChecker(linter))
