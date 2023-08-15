# pylint: disable=missing-function-docstring,missing-class-docstring
import general_superstaq as gss


def test_read_json_qubo_result() -> None:
    deserialized_return = [{0: 0, 1: 0, 2: 0, 3: 0, 4: 0}]

    server_return = {"solution": gss.serialization.serialize(deserialized_return)}

    assert repr(gss.qubo.read_json_qubo_result(server_return)) == repr(deserialized_return)
