import codecs


def bytes_to_str(bytes_data: bytes) -> str:
    return codecs.encode(bytes_data, "base64").decode()


def str_to_bytes(str_data: str) -> bytes:
    return codecs.decode(str_data.encode(), "base64")
