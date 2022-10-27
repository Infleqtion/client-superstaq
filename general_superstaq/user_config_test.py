import os
import secrets
import tempfile
from unittest import mock

import pytest

import general_superstaq as gss


def test_service_get_balance() -> None:
    client = gss.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="key", client_name="general_superstaq"
    )
    service = gss.user_config.UserConfig(client)
    mock_client = mock.MagicMock()
    mock_client.get_balance.return_value = {"balance": 12345.6789}
    service._client = mock_client

    assert service.get_balance() == "$12,345.68"
    assert service.get_balance(pretty_output=False) == 12345.6789


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaQClient.ibmq_set_token",
    return_value={"status": "Your IBMQ account token has been updated"},
)
def test_ibmq_set_token(mock_ibmq: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="key", client_name="general_superstaq"
    )
    service = gss.user_config.UserConfig(client)
    assert service.ibmq_set_token("valid token") == {
        "status": "Your IBMQ account token has been updated"
    }


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaQClient.aqt_upload_configs",
    return_value="Your AQT configuration has been updated",
)
def test_service_aqt_upload_configs(mock_aqt_compile: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="key", client_name="general_superstaq"
    )
    service = gss.user_config.UserConfig(client)
    tempdir = tempfile.gettempdir()
    pulses_file = os.path.join(tempdir, f"pulses-{secrets.token_hex(nbytes=16)}.yaml")
    variables_file = os.path.join(tempdir, f"variables-{secrets.token_hex(nbytes=16)}.yaml")

    with open(pulses_file, "w") as pulses:
        pulses.write("Hello")
    with open(variables_file, "w") as variables:
        variables.write("World")

    assert service.aqt_upload_configs(pulses_file, variables_file) == (
        "Your AQT configuration has been updated"
    )
    mock_aqt_compile.assert_called_with({"pulses": "Hello", "variables": "World"})

    assert service.aqt_upload_configs(pulses={"abc": 123}, variables={"xyz": "four"}) == (
        "Your AQT configuration has been updated"
    )
    mock_aqt_compile.assert_called_with({"pulses": "abc: 123\n", "variables": "xyz: four\n"})

    os.remove(variables_file)
    with pytest.raises(ValueError, match=r"variables-.*\.yaml' is not a valid file path"):
        _ = service.aqt_upload_configs(pulses_file, variables_file)

    os.remove(pulses_file)
    with pytest.raises(ValueError, match=r"pulses-.*\.yaml' is not a valid file path"):
        _ = service.aqt_upload_configs(pulses_file, variables_file)

    with pytest.raises(ValueError, match="AQT configs should be"):
        # Invalid input types:
        _ = service.aqt_upload_configs([], [])

    with pytest.raises(ValueError, match="AQT configs should be"):
        # Input type that can't be serialized with yaml.SafeDumper:
        _ = service.aqt_upload_configs({"foo": mock.DEFAULT}, {})

    with mock.patch.dict("sys.modules", {"yaml": None}):
        with pytest.raises(ModuleNotFoundError, match="PyYAML"):
            _ = service.aqt_upload_configs({}, {})


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaQClient.aqt_get_configs",
    return_value={"pulses": "Hello", "variables": "World"},
)
def test_service_aqt_get_configs(mock_aqt_compile: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="key", client_name="general_superstaq"
    )
    service = gss.user_config.UserConfig(client)
    tempdir = tempfile.gettempdir()
    pulses_file = secrets.token_hex(nbytes=16)
    variables_file = secrets.token_hex(nbytes=16)

    service.aqt_download_configs(
        f"{tempdir}/{pulses_file}.yaml", f"{tempdir}/{variables_file}.yaml"
    )

    with open(f"{tempdir}/{pulses_file}.yaml", "r") as file:
        assert file.read() == "Hello"

    with open(f"{tempdir}/{variables_file}.yaml", "r") as file:
        assert file.read() == "World"

    with pytest.raises(ValueError, match="exist."):
        service.aqt_download_configs(
            f"{tempdir}/{pulses_file}.yaml", f"{tempdir}/{variables_file}.yaml"
        )

    service.aqt_download_configs(
        f"{tempdir}/{pulses_file}.yaml", f"{tempdir}/{variables_file}.yaml", overwrite=True
    )

    with pytest.raises(ValueError, match="exists"):
        os.remove(f"{tempdir}/{pulses_file}.yaml")
        service.aqt_download_configs(
            f"{tempdir}/{pulses_file}.yaml", f"{tempdir}/{variables_file}.yaml"
        )

    os.remove(f"{tempdir}/{variables_file}.yaml")

    with pytest.raises(ValueError, match="exists"):
        service.aqt_download_configs(
            f"{tempdir}/{pulses_file}.yaml", f"{tempdir}/{variables_file}.yaml"
        )
        os.remove(f"{tempdir}/{variables_file}.yaml")
        service.aqt_download_configs(
            f"{tempdir}/{pulses_file}.yaml", f"{tempdir}/{variables_file}.yaml"
        )

    with pytest.raises(ValueError, match="Please provide both pulses and variables"):
        service.aqt_download_configs(variables_file_path="foo/bar.yaml")

    assert service.aqt_download_configs() == ("Hello", "World")

    with mock.patch.dict("sys.modules", {"yaml": None}):
        with pytest.raises(ModuleNotFoundError, match="PyYAML"):
            _ = service.aqt_download_configs()
