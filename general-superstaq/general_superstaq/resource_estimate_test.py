# pylint: disable=missing-function-docstring,missing-class-docstring
from general_superstaq import ResourceEstimate


def test_resource_estimate() -> None:
    json_data = {"num_single_qubit_gates": 1, "num_two_qubit_gates": 2, "depth": 3}
    expected_re = ResourceEstimate(1, 2, 3)
    constructed_re = ResourceEstimate(json_data=json_data)

    assert repr(expected_re) == repr(constructed_re)
