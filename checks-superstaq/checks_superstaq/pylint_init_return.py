"""Check return type annotation of __init__ functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astroid import Const, nodes
from pylint.checkers import BaseChecker
from pylint.checkers.utils import only_required_for_messages

if TYPE_CHECKING:
    from pylint.lint import PyLinter


class InitChecker(BaseChecker):
    """Checker for return type annotations for class constructors (__init__).

    * Check that all __init__ functions have the return type annotation '-> None'.
    """

    name = "init-return-check"
    msgs = {
        "W6061": (
            "Missing -> None return type annotation for __init__ method",
            "missing-return-type-annotation-init",
            "Missing type annotation for __init__ causes mypy check inconsistentcies.",
        ),
        "W6062": (
            "Incorrect return type annotation for __init__ method. Expected None but found '%s'",
            "incorrect-return-type-annotation-init",
            "__init__ functions should only have return type None.",
        ),
    }

    @only_required_for_messages("init-return-check")
    def visit_functiondef(self, node: nodes.FunctionDef) -> None:
        """Called for function and method definitions (def).

        Checks if the return type annotation for __init__ is explicity None or incorrect.

        Args:
            node: Node for a function or method definition in the AST.
        """
        if node.name == "__init__":
            if not node.returns:
                self.add_message("missing-return-type-annotation-init", node=node)
            elif not isinstance(node.returns, Const):
                self.add_message(
                    "incorrect-return-type-annotation-init",
                    args=node.returns.as_string(),
                    node=node,
                )


def register(linter: PyLinter) -> None:
    """Registers plugin to be accessed by pylint.

    Args:
        linter: The base pylinter which the custom checker will inherit from.
    """
    linter.register_checker(InitChecker(linter))
