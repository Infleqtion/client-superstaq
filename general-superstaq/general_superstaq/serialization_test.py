# pylint: disable=missing-function-docstring
import general_superstaq as gss


def test_serialization() -> None:
    obj = {"object": ["to", "serialize"]}
    serialized_obj = gss.serialization.serialize(obj)
    assert isinstance(serialized_obj, str)
    assert gss.serialization.deserialize(serialized_obj) == obj


def test_encrypt() -> None:
    token = "test_token"
    encrypted_token = gss.serialization.encrypt(token)
    assert isinstance(encrypted_token, str)
