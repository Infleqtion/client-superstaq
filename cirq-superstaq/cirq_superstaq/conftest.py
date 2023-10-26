import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register option to run hilbert marked tests
    Args:
        parser: The pytest parser from the command line
    """
    parser.addoption(
        "--run_hilbert_tests", action="store_true", default=False, help="runs Hilbert tests"
    )


def pytest_configure(config: pytest.Config) -> None:
    """Perform initial configuration
    Args:
        config: Passed in configuration
    """
    config.addinivalue_line("markers", "hilbert: mark test as Hilbert's")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Modify items in the collection

    Args:
        config: Passed in configuration
        items: Test functions
    """
    if config.getoption("--run_hilbert_tests"):
        # --run_hilbert_tests given in cli: do not skip Hilbert tests
        return
    skip_slow = pytest.mark.skip(reason="need --run_hilbert_tests option to run")
    for item in items:
        print(type(item))
        if "hilbert" in item.keywords:
            item.add_marker(skip_slow)
