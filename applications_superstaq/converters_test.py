import applications_superstaq


def test_serialization() -> None:
    obj = {"object": ["to", "serialize"]}
    serialized_obj = qiskit_superstaq.converters.serialize(obj)
    assert isinstance(serialized_obj, str)
    assert qiskit_superstaq.converters.deserialize(serialized_obj) == obj
