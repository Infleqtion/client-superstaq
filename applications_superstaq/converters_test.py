import pickle

import applications_superstaq


def test_bytes_str_conversions() -> None:
    str_data = applications_superstaq.converters.bytes_to_str(pickle.dumps("test_data"))
    assert isinstance(str_data, str)
    bytes_data = applications_superstaq.converters.str_to_bytes(str_data)
    assert pickle.loads(bytes_data) == "test_data"
