import packaging.version

import cirq_superstaq as css


def test_version() -> None:
    assert (
        packaging.version.Version("0.1.0")
        < packaging.version.parse(css.__version__)
        < packaging.version.Version("1.0.0")
    )
