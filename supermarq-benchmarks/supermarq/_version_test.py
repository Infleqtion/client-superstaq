# pylint: disable=missing-function-docstring,missing-class-docstring
import packaging.version

import supermarq


def test_version() -> None:
    assert (
        packaging.version.Version("0.0.4")
        < packaging.version.parse(supermarq.__version__)
        < packaging.version.Version("1.0.0")
    )
