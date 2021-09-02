import applications_superstaq


def test_serialization() -> None:
    obj = {"object": ["to", "serialize"]}
    serialized_obj = applications_superstaq.converters.serialize(obj)
    assert isinstance(serialized_obj, str)
    assert applications_superstaq.converters.deserialize(serialized_obj) == obj
