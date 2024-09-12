# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=missing-function-docstring,missing-class-docstring
from __future__ import annotations

import json
from unittest import mock

import cirq
import general_superstaq as gss
import pytest
import requests

import cirq_superstaq as css


@pytest.fixture
def job() -> css.Job:
    """Fixture for cirq_superstaq Job.

    Returns:
        A `cirq_superstaq` Job instance.
    """
    client = gss.superstaq_client._SuperstaqClient(
        client_name="cirq-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    return css.Job(client, "job_id")


@pytest.fixture
def job_dict() -> dict[str, object]:
    """Fixture for a standard, completed single job result.

    Returns:
        A dictionary containing commonly expected job data.
    """
    return {
        "data": {"histogram": {"11": 1}},
        "num_qubits": 2,
        "samples": {"11": 1},
        "shots": 1,
        "status": "Done",
        "target": "ss_unconstrained_simulator",
    }


def modifiy_job_result(base_result: dict[str, object], **kwargs: object) -> dict[str, object]:
    """Extends and updates `base_result` with passed `kwargs`.

    Args:
        base_result: Base job result to modify.
        kwargs: Additional keyword args to add or update existing values in `base_result`.

    Returns:
        An updated dictionary based on the `kwargs` passed.
    """
    return {**base_result, **kwargs}


def new_job() -> css.Job:
    """Creates a new `cirq_superstaq` Job instance.

    Returns:
        A `cirq_superstaq` Job instance.
    """
    client = gss.superstaq_client._SuperstaqClient(
        client_name="cirq-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    return css.Job(client, "new_job_id")


@pytest.fixture
def multi_circuit_job() -> css.Job:
    """Fixture for a job with multiple circuits submitted

    Returns:
        A job with multiple subjobs
    """
    client = gss.superstaq_client._SuperstaqClient(
        client_name="cirq-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    job = css.Job(client, "job_id1,job_id2,job_id3")
    job._job = {
        "job_id1": {"status": "Done"},
        "job_id2": {"status": "Running"},
        "job_id3": {"status": "Submitted"},
    }
    return job


def _mocked_request_response(content: object) -> requests.Response:
    response = requests.Response()
    response.status_code = requests.codes.OK
    response._content = json.dumps(content).encode()
    return response


def patched_requests(*contents: object) -> mock._patch[mock.Mock]:
    """Mocks the server's response given sequence of content objects.

    Args:
        contents: The JSON contents to return for each request.

    Returns:
        A mock patch that returns the provided content.
    """
    responses = [_mocked_request_response(val) for val in contents]
    return mock.patch("requests.Session.post", side_effect=responses)


def test_cancel(job: css.Job, multi_circuit_job: css.Job) -> None:
    with mock.patch("requests.Session.post", return_value=mock.MagicMock(ok=True)) as mock_post:
        job.cancel()
        new_job().cancel()
        multi_circuit_job.cancel(index=2)
        assert mock_post.call_count == 3


def test_job_fields(job: css.Job, job_dict: dict[str, object]) -> None:
    compiled_circuit = cirq.Circuit(cirq.H(cirq.q(0)), cirq.measure(cirq.q(0)))
    job_result = modifiy_job_result(
        job_dict, compiled_circuit=css.serialize_circuits(compiled_circuit)
    )

    assert job.job_id() == "job_id"

    with patched_requests({"job_id": job_result}) as mocked_get_job:
        assert job.target() == "ss_unconstrained_simulator"
        assert job.num_qubits(index=0) == 2
        assert job.repetitions() == 1
        assert job.compiled_circuits(index=0) == compiled_circuit
        mocked_get_job.assert_called_once()  # Only refreshed once


def test_target(job: css.Job, job_dict: dict[str, object]) -> None:
    # The first call will trigger a refresh:
    with patched_requests({"job_id": job_dict}) as mocked_get_job:
        assert job.target() == "ss_unconstrained_simulator"
        mocked_get_job.assert_called_once()

    # Shouldn't need to retrieve anything now that `job._job` is populated:
    assert job.target() == "ss_unconstrained_simulator"


def test_num_qubits(job: css.Job, job_dict: dict[str, object]) -> None:
    # The first call will trigger a refresh:
    with patched_requests({"job_id": job_dict}) as mocked_get_job:
        # Test case: index -> int
        assert job.num_qubits(index=0) == 2
        mocked_get_job.assert_called_once()

    # Shouldn't need to retrieve anything now that `job._job` is populated:
    assert job.num_qubits(index=0) == 2
    assert job.num_qubits() == [2]


def test_repetitions(job: css.Job, job_dict: dict[str, object]) -> None:
    # The first call will trigger a refresh:
    with patched_requests({"job_id": job_dict}) as mocked_get_job:
        assert job.repetitions() == 1
        mocked_get_job.assert_called_once()

    # Shouldn't need to retrieve anything now that `job._job` is populated:
    assert job.repetitions() == 1


def test_get_circuit(job: css.Job) -> None:
    with pytest.raises(ValueError, match="The circuit type requested is invalid."):
        job._get_circuits("invalid_type")


def test_compiled_circuit(job: css.Job) -> None:
    compiled_circuit = cirq.Circuit(cirq.H(cirq.q(0)), cirq.measure(cirq.q(0)))
    job_dict = {
        "status": "Done",
        "target": "fake_target",
        "compiled_circuit": css.serialize_circuits(compiled_circuit),
    }

    # The first call will trigger a refresh:
    with patched_requests({"job_id": job_dict}) as mocked_get_job:
        assert job.compiled_circuits(index=0) == compiled_circuit
        mocked_get_job.assert_called_once()

    # Shouldn't need to retrieve anything now that `job._job` is populated:
    assert job.compiled_circuits(index=0) == compiled_circuit

    with pytest.raises(
        ValueError, match=f"Target '{job.target()}' does not use pulse gate circuits."
    ):
        job.pulse_gate_circuits()


def test_pulse_gate_circuits(job: css.Job) -> None:
    import qiskit

    qss = pytest.importorskip("qiskit_superstaq", reason="qiskit-superstaq is not installed")
    pulse_gate_circuit = qiskit.QuantumCircuit(1, 1)
    pulse_gate = qiskit.circuit.Gate("test_pulse_gate", 1, [3.14, 1])
    pulse_gate_circuit.append(pulse_gate, [0])
    pulse_gate_circuit.measure(0, 0)

    job_dict = {"status": "Done", "pulse_gate_circuits": qss.serialize_circuits(pulse_gate_circuit)}

    # The first call will trigger a refresh:
    with patched_requests({"job_id": job_dict}) as mocked_get_job:
        assert job.pulse_gate_circuits()[0] == pulse_gate_circuit
        mocked_get_job.assert_called_once()

    # Shouldn't need to retrieve anything now that `job._job` is populated:
    assert job.pulse_gate_circuits()[0] == pulse_gate_circuit


def test_pulse_gate_circuits_index(job: css.Job) -> None:
    import qiskit

    qss = pytest.importorskip("qiskit_superstaq", reason="qiskit-superstaq is not installed")
    pulse_gate_circuit = qiskit.QuantumCircuit(1, 1)
    pulse_gate = qiskit.circuit.Gate("test_pulse_gate", 1, [3.14, 1])
    pulse_gate_circuit.append(pulse_gate, [0])
    pulse_gate_circuit.measure(0, 0)

    job_dict = {"status": "Done", "pulse_gate_circuits": qss.serialize_circuits(pulse_gate_circuit)}

    # The first call will trigger a refresh:
    with patched_requests({"job_id": job_dict}) as mocked_get_job:
        assert job.pulse_gate_circuits(index=0)[0] == pulse_gate_circuit
        mocked_get_job.assert_called_once()

    # Shouldn't need to retrieve anything now that `job._job` is populated:
    assert job.pulse_gate_circuits(index=0)[0] == pulse_gate_circuit

    # Test on invalid index
    with pytest.raises(ValueError, match="is less than the minimum"):
        job.pulse_gate_circuits(index=-3)


def test_pulse_gate_circuits_invalid_circuit(job: css.Job) -> None:
    # Invalid pulse gate circuit

    job_dict = {"status": "Done", "pulse_gate_circuits": "invalid_pulse_gate_circuit_str"}

    # The first call will trigger a refresh:
    with patched_requests({"job_id": job_dict}):
        with pytest.raises(ValueError, match="circuits could not be deserialized."), pytest.warns(
            match="pulse gate circuit could not be deserialized"
        ):
            job.pulse_gate_circuits()


def test_multi_pulse_gate_circuits(multi_circuit_job: css.Job) -> None:
    import qiskit

    qss = pytest.importorskip("qiskit_superstaq", reason="qiskit-superstaq is not installed")
    pulse_gate_circuit = qiskit.QuantumCircuit(1, 1)
    pulse_gate = qiskit.circuit.Gate("test_pulse_gate", 1, [3.14, 1])
    pulse_gate_circuit.append(pulse_gate, [0])
    pulse_gate_circuit.measure(0, 0)

    job_dict = {
        "status": "Done",
        "pulse_gate_circuits": qss.serialize_circuits(pulse_gate_circuit),
    }

    job_info = {f"job_id{i}": job_dict for i in range(1, 4)}
    multi_circuit_job._job = job_info

    with patched_requests(job_info):
        assert multi_circuit_job.pulse_gate_circuits() == [
            pulse_gate_circuit,
            pulse_gate_circuit,
            pulse_gate_circuit,
        ]


def test_multi_circuit_job(multi_circuit_job: css.Job) -> None:
    compiled_circuit = cirq.Circuit(
        cirq.H(cirq.q(0)),
        cirq.CNOT(cirq.q(2), cirq.q(0)),
        cirq.X(cirq.q(1)) ** 0.5,
        cirq.measure(cirq.q(0), cirq.q(1), cirq.q(2)),
    )
    job_dict = {
        "status": "Done",
        "num_qubits": 3,
        "data": {"histogram": {"000": 0.16, "010": 0.36, "100": 0.3, "110": 0.18}},
        "samples": {"000": 8, "010": 18, "100": 15, "110": 9},
        "shots": 50,
        "compiled_circuit": css.serialize_circuits(compiled_circuit),
        "input_circuit": css.serialize_circuits(compiled_circuit),
    }
    job_info = {f"job_id{i}": job_dict for i in range(1, 4)}
    multi_circuit_job._job = job_info

    with patched_requests(job_info):
        # Test case: No index
        assert multi_circuit_job.compiled_circuits() == [
            compiled_circuit,
            compiled_circuit,
            compiled_circuit,
        ]
        assert multi_circuit_job.num_qubits(index=None) == [3, 3, 3]
        assert multi_circuit_job.counts() == [
            {"000": 8, "010": 18, "100": 15, "110": 9},
            {"000": 8, "010": 18, "100": 15, "110": 9},
            {"000": 8, "010": 18, "100": 15, "110": 9},
        ]
        assert multi_circuit_job.counts(qubit_indices=[0]) == [
            {"0": 26, "1": 24},
            {"0": 26, "1": 24},
            {"0": 26, "1": 24},
        ]

        # Test case: with index
        assert multi_circuit_job.compiled_circuits(index=2) == compiled_circuit
        assert multi_circuit_job.num_qubits(index=2) == 3
        assert multi_circuit_job.counts(index=2) == {"000": 8, "010": 18, "100": 15, "110": 9}
        assert multi_circuit_job.counts(index=2, qubit_indices=[0]) == {"0": 26, "1": 24}


def test_input_circuit(job: css.Job) -> None:
    input_circuit = cirq.Circuit(cirq.H(cirq.q(0)), cirq.measure(cirq.q(0)))
    job_dict = {
        "status": "Done",
        "input_circuit": css.serialize_circuits(input_circuit),
        "compiled_circuit": css.serialize_circuits(input_circuit),
    }

    # The first call will trigger a refresh:
    with patched_requests({"job_id": job_dict}) as mocked_get_job:
        assert job.input_circuits(index=0) == input_circuit
        mocked_get_job.assert_called_once()

    # Shouldn't need to retrieve anything now that `job._job` is populated:
    assert job.input_circuits(index=0) == input_circuit


def test_job_status_refresh() -> None:
    completed_job_dict = {"new_job_id": {"status": "Done"}}

    for status in css.Job.NON_TERMINAL_STATES:
        job_dict = {"new_job_id": {"status": status}}

        job = new_job()
        with patched_requests(job_dict, completed_job_dict) as mocked_request:
            assert job.status() == status
            assert job.status() == "Done"
            assert mocked_request.call_count == 2
            assert mocked_request.call_args.kwargs["json"] == {"job_ids": ["new_job_id"]}

    for status in css.Job.TERMINAL_STATES:
        job_dict = {"new_job_id": {"status": status}}

        job = new_job()
        with patched_requests(job_dict, completed_job_dict) as mocked_request:
            assert job.status() == status
            assert job.status() == status
            mocked_request.assert_called_once()
            assert mocked_request.call_args.kwargs["json"] == {"job_ids": ["new_job_id"]}


def test_value_equality(job: css.Job) -> None:
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(job, job)
    eq.add_equality_group(new_job())


def test_job_str_repr_eq(job: css.Job) -> None:
    assert str(job) == "Job with job_id=job_id"
    cirq.testing.assert_equivalent_repr(
        job, setup_code="import cirq_superstaq as css\nimport general_superstaq as gss"
    )

    assert not job == 1


def test_job_to_dict(job: css.Job, job_dict: dict[str, object]) -> None:
    job_result = {"job_id": job_dict}
    job._job = {}
    with patched_requests(job_result):
        assert job.to_dict() == job_result


def test_job_counts(job: css.Job, job_dict: dict[str, object]) -> None:
    job_result = modifiy_job_result(job_dict, data={"histogram": {"10": 1}}, samples={"10": 1})
    with patched_requests({"job_id": job_result}):
        assert job.counts(index=0) == {"10": 1}
        assert job.counts(index=0, qubit_indices=[0]) == ({"1": 1})
        assert job.counts() == [{"10": 1}]


def test_job_counts_failed(job: css.Job, job_dict: dict[str, object]) -> None:
    job_result = modifiy_job_result(job_dict, status="Failed", failure={"error": "too many qubits"})
    with patched_requests({"job_id": job_result}):
        with pytest.raises(gss.SuperstaqUnsuccessfulJobException, match="too many qubits"):
            _ = job.counts()
        assert job.status() == "Failed"


@mock.patch("time.sleep", return_value=None)
def test_job_counts_poll(
    mock_sleep: mock.MagicMock, job: css.Job, job_dict: dict[str, object]
) -> None:
    ready_job = {
        "status": "Ready",
    }
    with patched_requests({"job_id": ready_job}, {"job_id": job_dict}) as mocked_requests:
        results = job.counts(index=0, polling_seconds=0)
        assert results == {"11": 1}
        assert mocked_requests.call_count == 2
        mock_sleep.assert_called_once()


@mock.patch("time.sleep", return_value=None)
@mock.patch("time.time", side_effect=range(20))
def test_job_counts_poll_timeout(
    _mock_time: mock.MagicMock, mock_sleep: mock.MagicMock, job: css.job.Job
) -> None:
    ready_job = {
        "status": "Ready",
    }
    with patched_requests(*[{"job_id": ready_job}] * 20):
        with pytest.raises(TimeoutError, match="Ready"):
            _ = job.counts(timeout_seconds=1, polling_seconds=0.1)
    assert mock_sleep.call_count == 11


@mock.patch("time.sleep", return_value=None)
def test_job_results_poll_failure(mock_sleep: mock.MagicMock, job: css.Job) -> None:
    running_job = {
        "status": "Running",
    }
    failed_job = {
        "status": "Failed",
        "failure": {"error": "too many qubits"},
    }

    with patched_requests(*[{"job_id": running_job}] * 5, {"job_id": failed_job}):
        with pytest.raises(gss.SuperstaqUnsuccessfulJobException, match="too many qubits"):
            _ = job.counts(timeout_seconds=1, polling_seconds=0.1)
    assert mock_sleep.call_count == 5


@mock.patch("time.sleep", return_value=None)
def test_status_with_index(
    mock_sleep: mock.MagicMock, multi_circuit_job: css.Job, job_dict: dict[str, int]
) -> None:
    running_job = {
        "status": "Running",
    }
    failed_job = {
        "status": "Failed",
        "failure": {"error": "too many qubits"},
    }

    job_info = {"job_id1": running_job, "job_id2": job_dict, "job_id3": failed_job}
    multi_circuit_job._job = job_info

    with patched_requests(job_info):
        # Overall status
        assert multi_circuit_job.status() == "Running"

        # Specific job status
        assert multi_circuit_job.status(index=1) == "Done"
        assert multi_circuit_job.counts(index=1, timeout_seconds=1, polling_seconds=0.5) == {
            "11": 1
        }
        assert mock_sleep.call_count == 0


def test_job_getitem(multi_circuit_job: css.Job) -> None:
    job_1 = multi_circuit_job[0]
    assert isinstance(job_1, css.Job)
    assert job_1.job_id() == "job_id1"
    assert job_1.status() == "Done"


def test_get_marginal_counts() -> None:
    counts_dict = {"10": 50, "11": 50}
    indices = [0]
    assert css.job._get_marginal_counts(counts_dict, indices) == ({"1": 100})
