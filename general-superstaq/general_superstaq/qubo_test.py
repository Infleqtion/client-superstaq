# pylint: disable=missing-function-docstring,missing-class-docstring
import qubovert as qv

import general_superstaq as gss


def test_read_json_qubo_result() -> None:

    deserialized_return = [{0: 0, 1: 0, 2: 0, 3: 0, 4: 0}]

    server_return = {"solution": gss.serialization.serialize(deserialized_return)}

    assert repr(gss.qubo.read_json_qubo_result(server_return)) == repr(deserialized_return)


def test_convert_qubo_to_model() -> None:
    example_qubo = qv.QUBO({(0,): 1.0, (1,): 1.0, (0, 1): -2.0})
    qubo_model = [
        {"keys": ["0"], "value": 1.0},
        {"keys": ["1"], "value": 1.0},
        {"keys": ["0", "1"], "value": -2.0},
    ]
    assert gss.qubo.convert_qubo_to_model(example_qubo) == qubo_model
