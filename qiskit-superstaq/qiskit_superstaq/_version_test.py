# pylint: disable=missing-function-docstring,missing-class-docstring
import packaging.version

import qiskit_superstaq as qss


def test_version() -> None:
    assert (
        packaging.version.Version("0.1.0")
        < packaging.version.parse(qss.__version__)
        < packaging.version.Version("1.0.0")
    )
