import packaging.version

import supermarq as sm


def test_version() -> None:
    assert (
        packaging.version.Version("0.0.4")
        < packaging.version.parse(sm.__version__)
        < packaging.version.Version("1.0.0")
    )
