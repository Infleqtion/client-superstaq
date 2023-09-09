# pylint: disable=missing-function-docstring,missing-class-docstring
import os
import secrets
import tempfile
from unittest import mock

import pytest
import qubovert as qv

import general_superstaq as gss


def test_service_get_balance() -> None:
    service = gss.service.Service(remote_host="http://example.com", api_key="key")
    mock_client = mock.MagicMock()
    mock_client.get_balance.return_value = {"balance": 12345.6789}
    service._client = mock_client

    assert service.get_balance() == "$12,345.68"
    assert service.get_balance(pretty_output=False) == 12345.6789


def test_accept_terms_of_use() -> None:
    service = gss.service.Service(remote_host="http://example.com", api_key="key")
    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaqClient.post_request"
    ) as mock_post_request:
        service._accept_terms_of_use("response")
        mock_post_request.assert_called_once_with(
            "/accept_terms_of_use", {"user_input": "response"}
        )


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaqClient.post_request",
    return_value="The user has been added",
)
def test_add_new_user(
    mock_post_request: mock.MagicMock,
) -> None:
    service = gss.service.Service(remote_host="http://example.com", api_key="key")
    assert service.add_new_user("Marie Curie", "mc@gmail.com") == "The user has been added"


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaqClient.post_request",
    return_value="The account's balance has been updated",
)
def test_update_user_balance(
    mock_post_request: mock.MagicMock,
) -> None:
    service = gss.service.Service(remote_host="http://example.com", api_key="key")
    assert (
        service.update_user_balance("mc@gmail.com", 5.00)
        == "The account's balance has been updated"
    )


def test_update_user_balance_limit() -> None:
    service = gss.service.Service(remote_host="http://example.com", api_key="key")
    with pytest.raises(gss.SuperstaqException, match="exceeds limit."):
        (service.update_user_balance("mc@gmail.com", 3500.00))


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaqClient.post_request",
    return_value="The account's role has been updated",
)
def test_update_user_role(
    mock_post_request: mock.MagicMock,
) -> None:
    service = gss.service.Service(remote_host="http://example.com", api_key="key")
    assert service.update_user_role("mc@gmail.com", 5) == "The account's role has been updated"


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaqClient.post_request",
    return_value={
        "qubo": [
            {"keys": ["0"], "value": 1.0},
            {"keys": ["1"], "value": 1.0},
            {"keys": ["0", "1"], "value": -2.0},
        ],
        "target": "ss_example_qpu",
        "shots": 10,
        "method": "dry-run",
        "max_solutions": 13,
    },
)
def test_submit_qubo(
    mock_post_request: mock.MagicMock,
) -> None:
    example_qubo = qv.QUBO({(0,): 1.0, (1,): 1.0, (0, 1): -2.0})
    target = "ss_example_qpu"
    repetitions = 10

    service = gss.service.Service(remote_host="http://example.com", api_key="key")
    assert service.submit_qubo(example_qubo, target, repetitions=repetitions, method="dry-run") == {
        "qubo": [
            {"keys": ["0"], "value": 1.0},
            {"keys": ["1"], "value": 1.0},
            {"keys": ["0", "1"], "value": -2.0},
        ],
        "target": "ss_example_qpu",
        "shots": 10,
        "method": "dry-run",
        "max_solutions": 13,
    }


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaqClient.aqt_upload_configs",
    return_value="Your AQT configuration has been updated",
)
def test_service_aqt_upload_configs(
    mock_aqt_compile: mock.MagicMock,
) -> None:
    service = gss.service.Service(remote_host="http://example.com", api_key="key")
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
    "general_superstaq.superstaq_client._SuperstaqClient.aqt_get_configs",
    return_value={"pulses": "Hello", "variables": "World"},
)
def test_service_aqt_get_configs(
    mock_aqt_compile: mock.MagicMock,
) -> None:
    service = gss.service.Service(remote_host="http://example.com", api_key="key")
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


@mock.patch("requests.post")
def test_aces(
    mock_post: mock.MagicMock,
) -> None:
    service = gss.service.Service(remote_host="http://example.com", api_key="key")
    mock_post.return_value.json = lambda: "id1"
    assert (
        service.submit_aces(
            target="ss_unconstrained_simulator",
            qubits=[0, 1],
            shots=100,
            num_circuits=10,
            mirror_depth=5,
            extra_depth=5,
            noise="bit_flip",
            error_prob=0.1,
        )
        == "id1"
    )

    mock_post.return_value.json = lambda: [1] * 51
    assert service.process_aces("id1") == [1] * 51
