import general_superstaq as gss


def test_serialization() -> None:
    obj = {"object": ["to", "serialize"]}
    serialized_obj = gss.serialization.serialize(obj)
    assert isinstance(serialized_obj, str)
    assert gss.serialization.deserialize(serialized_obj) == obj
