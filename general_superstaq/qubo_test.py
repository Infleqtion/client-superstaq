import numpy as np
import qubovert as qv

import general_superstaq as gss


def test_read_json_qubo_result() -> None:
    example_solution = np.rec.fromrecords(
        [({0: 0, 1: 1, 3: 1}, -1, 6), ({0: 1, 1: 1, 3: 1}, -1, 4)],
        dtype=[("solution", "O"), ("energy", "<f8"), ("num_occurrences", "<i8")],
    )
    json_dict = {
        "solution": gss.serialization.serialize(example_solution),
    }
    assert repr(gss.qubo.read_json_qubo_result(json_dict)) == repr(example_solution)


def test_convert_qubo_to_model() -> None:
    example_qubo = qv.QUBO({(0,): 1.0, (1,): 1.0, (0, 1): -2.0})
    qubo_model = [
        {"keys": ["0"], "value": 1.0},
        {"keys": ["1"], "value": 1.0},
        {"keys": ["0", "1"], "value": -2.0},
    ]
    assert gss.qubo.convert_qubo_to_model(example_qubo) == qubo_model
