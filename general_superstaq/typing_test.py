import importlib
import typing
from unittest import mock

import typing_extensions

import general_superstaq as gss


def test_import() -> None:
    """Get full coverage of the conditional `TypedDict` import in gss.typing."""

    with mock.patch("sys.version_info", (3, 7)):
        importlib.reload(gss.typing)
        assert gss.typing.TypedDict is typing_extensions.TypedDict

    setattr(typing, "TypedDict", typing_extensions.TypedDict)  # So this passes on python 3.7
    setattr(typing_extensions, "TypedDict", None)

    with mock.patch("sys.version_info", (3, 8)):
        importlib.reload(gss.typing)
        assert gss.typing.TypedDict is getattr(typing, "TypedDict")
        assert gss.typing.TypedDict is not typing_extensions.TypedDict
