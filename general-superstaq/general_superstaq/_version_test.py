# pylint: disable=missing-function-docstring,missing-class-docstring
import packaging.version

import general_superstaq as gss


def test_version() -> None:
    assert (
        packaging.version.Version("0.1.0")
        < packaging.version.parse(gss.__version__)
        < packaging.version.Version("1.0.0")
    )
