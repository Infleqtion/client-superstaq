import general_superstaq as gss


def test_serialization() -> None:
    obj = {"object": ["to", "serialize"]}
    serialized_obj = gss.converters.serialize(obj)
    assert isinstance(serialized_obj, str)
    assert gss.converters.deserialize(serialized_obj) == obj
