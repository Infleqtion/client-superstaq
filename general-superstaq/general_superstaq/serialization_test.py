# pylint: disable=missing-function-docstring
from unittest import mock

import rsa

import general_superstaq as gss


def test_serialization() -> None:
    obj = {"object": ["to", "serialize"]}
    serialized_obj = gss.serialization.serialize(obj)
    assert isinstance(serialized_obj, str)
    assert gss.serialization.deserialize(serialized_obj) == obj


def test_encrypt() -> None:
    public, private = rsa.newkeys(512)
    with mock.patch("general_superstaq.TOKEN_PUBLIC_KEY_E", public.e), mock.patch(
        "general_superstaq.TOKEN_PUBLIC_KEY_N", public.n
    ):
        val = "abc123"
        encrypted_val = gss.serialization.encrypt(val)
        assert isinstance(encrypted_val, str)
        assert rsa.decrypt(gss.serialization._str_to_bytes(encrypted_val), private).decode() == val
