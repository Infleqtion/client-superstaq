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
# pylint: disable=missing-function-docstring
# mypy: disable-error-code=method-assign

from __future__ import annotations

import os
from typing import NamedTuple
from unittest.mock import MagicMock, call, patch

import cirq
import numpy as np
import pandas as pd
import pytest
from general_superstaq.superstaq_exceptions import SuperstaqServerException

from supermarq.qcvv.base_experiment import BenchmarkingExperiment, Sample


@pytest.fixture(scope="session", autouse=True)
def patch_tqdm() -> None:
    os.environ["TQDM_DISABLE"] = "1"


@pytest.fixture
@patch.multiple(BenchmarkingExperiment, __abstractmethods__=set())
def abc_experiment() -> BenchmarkingExperiment:
    with patch("cirq_superstaq.service.Service"):
        return BenchmarkingExperiment(num_qubits=2)  # type: ignore[abstract]


@pytest.fixture
def sample_circuits() -> list[Sample]:
    qubits = cirq.LineQubit.range(2)
    return [
        Sample(
            circuit=cirq.Circuit(cirq.CZ(*qubits), cirq.MeasurementGate(num_qubits=2)(*qubits)),
            data={"circuit": 1},
        ),
        Sample(circuit=cirq.Circuit(cirq.CX(*qubits)), data={"circuit": 2}),
    ]


class ExampleResults(NamedTuple):
    """NamedTuple instance to use for testing"""

    example: float


def test_benchmarking_experiment_init(abc_experiment: BenchmarkingExperiment) -> None:
    assert abc_experiment.num_qubits == 2
    assert abc_experiment._raw_data is None
    assert abc_experiment._results is None
    assert abc_experiment._samples is None

    abc_experiment._raw_data = pd.DataFrame([{"Example": 0.1}])
    abc_experiment._results = ExampleResults(example=5.0)

    pd.testing.assert_frame_equal(abc_experiment.raw_data, abc_experiment._raw_data)
    assert abc_experiment.results == abc_experiment._results


def test_empty_results_error(abc_experiment: BenchmarkingExperiment) -> None:
    with pytest.raises(
        RuntimeError, match="No results to retrieve. The experiment has not been run."
    ):
        _ = abc_experiment.results


def test_empty_data_error(abc_experiment: BenchmarkingExperiment) -> None:
    with pytest.raises(RuntimeError, match="No data to retrieve. The experiment has not been run."):
        _ = abc_experiment.raw_data


def test_empty_samples_error(abc_experiment: BenchmarkingExperiment) -> None:
    with pytest.raises(
        RuntimeError, match="No samples to retrieve. The experiment has not been run."
    ):
        _ = abc_experiment.samples


def test_run_results_overwrite_warning(abc_experiment: BenchmarkingExperiment) -> None:
    abc_experiment._samples = [Sample(circuit=MagicMock(), data={})]
    abc_experiment.build_circuits = MagicMock()
    abc_experiment.sample_circuits_with_simulator = MagicMock()
    abc_experiment.process_probabilities = MagicMock()

    print(abc_experiment._results)
    with pytest.warns(UserWarning, match="Existing results will be overwritten."):
        abc_experiment.run(100, [1, 50, 100])


def test_run_with_bad_layers(abc_experiment: BenchmarkingExperiment) -> None:
    with pytest.raises(ValueError, match="The `layers` iterator can only include positive values."):
        abc_experiment.run(20, [0])


def test_run_local(abc_experiment: BenchmarkingExperiment) -> None:
    abc_experiment.build_circuits = (mock_build_circuits := MagicMock())
    abc_experiment.submit_ss_jobs = (mock_submit_ss_jobs := MagicMock())
    abc_experiment.sample_circuits_with_simulator = (
        mock_sample_circuits_with_simulator := MagicMock()
    )
    abc_experiment.run(50, [1, 50, 100], shots=50)

    mock_build_circuits.assert_called_once_with(50, [1, 50, 100])

    mock_sample_circuits_with_simulator.assert_called_once()
    call_args = mock_sample_circuits_with_simulator.call_args_list[0][0]
    assert call_args[1] == 50
    assert isinstance(call_args[0], cirq.Simulator)  # Test simulated on a default target

    mock_submit_ss_jobs.assert_not_called()


def test_run_local_defined_sim(abc_experiment: BenchmarkingExperiment) -> None:
    abc_experiment.build_circuits = (mock_build_circuits := MagicMock())
    abc_experiment.submit_ss_jobs = (mock_submit_ss_jobs := MagicMock())
    abc_experiment.sample_circuits_with_simulator = (
        mock_sample_circuits_with_simulator := MagicMock()
    )

    abc_experiment.run(
        50, [1, 50, 100], shots=50, target=(target_sim := cirq.DensityMatrixSimulator())
    )

    mock_build_circuits.assert_called_once_with(50, [1, 50, 100])

    mock_sample_circuits_with_simulator.assert_called_once()
    call_args = mock_sample_circuits_with_simulator.call_args_list[0][0]
    assert call_args[1] == 50
    assert call_args[0] == target_sim  # Test simulated on the given target

    mock_submit_ss_jobs.assert_not_called()


def test_run_on_ss_server(abc_experiment: BenchmarkingExperiment) -> None:
    abc_experiment.build_circuits = (mock_build_circuits := MagicMock())
    abc_experiment.submit_ss_jobs = (mock_submit_ss_jobs := MagicMock())
    abc_experiment.sample_circuits_with_simulator = (
        mock_sample_circuits_with_simulator := MagicMock()
    )
    abc_experiment.run(
        50, [1, 50, 100], shots=50, target="example_ss_target", target_options={"some": "options"}
    )

    mock_build_circuits.assert_called_once_with(50, [1, 50, 100])

    mock_sample_circuits_with_simulator.assert_not_called()

    mock_submit_ss_jobs.assert_called_once_with(
        "example_ss_target", 50, None, **{"some": "options"}
    )


def test_run_with_simulator(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment._samples = sample_circuits
    test_target = MagicMock(spec=cirq.Simulator)
    test_target.sample = MagicMock(
        return_value=MagicMock(values=np.ones(shape=(100, 1), dtype=np.int64))
    )

    abc_experiment.sample_circuits_with_simulator(test_target, shots=100)

    # Test simulator calls
    test_target.sample.assert_has_calls(
        [
            call(sample_circuits[0].circuit, repetitions=100),
            call(sample_circuits[1].circuit, repetitions=100),
        ]
    )

    # Test probabilities
    assert sample_circuits[0].probabilities == {"00": 0.0, "01": 1.0, "10": 0.0, "11": 0.0}
    assert sample_circuits[1].probabilities == {"00": 0.0, "01": 1.0, "10": 0.0, "11": 0.0}


def test_submit_ss_jobs(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:

    abc_experiment._samples = sample_circuits
    abc_experiment._service = (mock_service := MagicMock())
    mock_service().create_job.side_effect = [MagicMock(job_id="job_1"), MagicMock(job_id="job_2")]

    mock_service().target_info.return_value = {}
    abc_experiment.submit_ss_jobs("example_target", shots=100)

    mock_service.create_job.assert_has_calls(
        [
            call(
                sample_circuits[0].circuit,
                target="example_target",
                method=None,
                repetitions=100,
            ),
            call(
                sample_circuits[1].circuit,
                target="example_target",
                method=None,
                repetitions=100,
            ),
        ],
        any_order=True,
    )


def test_submit_ss_jobs_dry_run(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:

    abc_experiment._samples = sample_circuits
    abc_experiment._service = (mock_service := MagicMock())
    mock_service().create_job.side_effect = [MagicMock(job_id="job_1"), MagicMock(job_id="job_2")]

    mock_service().target_info.return_value = {}
    abc_experiment.submit_ss_jobs("example_target", shots=100, method="dry-run")

    mock_service.create_job.assert_has_calls(
        [
            call(
                sample_circuits[0].circuit,
                target="example_target",
                method="dry-run",
                repetitions=100,
            ),
            call(
                sample_circuits[1].circuit,
                target="example_target",
                method="dry-run",
                repetitions=100,
            ),
        ],
        any_order=True,
    )


def test_submit_ss_jobs_job_already_has_id(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:

    abc_experiment._samples = sample_circuits
    sample_circuits[0].job = "example_job_id"
    abc_experiment._service = (mock_service := MagicMock())
    mock_service().create_job.side_effect = [MagicMock(job_id="job_1"), MagicMock(job_id="job_2")]

    mock_service().target_info.return_value = {}
    abc_experiment.submit_ss_jobs("example_target", shots=100, method="example_method")

    mock_service.create_job.assert_has_calls(
        [
            call(
                sample_circuits[1].circuit,
                target="example_target",
                method="example_method",
                repetitions=100,
            ),
        ],
        any_order=True,
    )


def test_submit_ss_jobs_with_exception(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment._samples = sample_circuits
    abc_experiment._service = (mock_service := MagicMock())

    mock_service.create_job.side_effect = SuperstaqServerException("example_exception")
    with pytest.warns(UserWarning):
        abc_experiment.submit_ss_jobs("example_target", shots=100)


def test_state_probs_to_dict(abc_experiment: BenchmarkingExperiment) -> None:
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    out_dict = abc_experiment._state_probs_to_dict(probabilities)
    assert out_dict == {
        "00": 0.1,
        "01": 0.2,
        "10": 0.3,
        "11": 0.4,
    }


def test_interleave_circuit() -> None:
    qubit = cirq.LineQubit(0)
    circuit = cirq.Circuit(*[cirq.X(qubit) for _ in range(4)])

    # With last gate
    interleaved_circuit = BenchmarkingExperiment._interleave_gate(
        circuit, cirq.Z, include_final=True
    )
    cirq.testing.assert_same_circuits(
        interleaved_circuit,
        cirq.Circuit(
            cirq.X(qubit),
            cirq.Z(qubit),
            cirq.X(qubit),
            cirq.Z(qubit),
            cirq.X(qubit),
            cirq.Z(qubit),
            cirq.X(qubit),
            cirq.Z(qubit),
        ),
    )

    # Without last gate
    interleaved_circuit = BenchmarkingExperiment._interleave_gate(
        circuit, cirq.Z, include_final=False
    )
    cirq.testing.assert_same_circuits(
        interleaved_circuit,
        cirq.Circuit(
            cirq.X(qubit),
            cirq.Z(qubit),
            cirq.X(qubit),
            cirq.Z(qubit),
            cirq.X(qubit),
            cirq.Z(qubit),
            cirq.X(qubit),
        ),
    )


def test_sample_statuses(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment._samples = sample_circuits
    sample_circuits[0].job = "example_job_id"
    abc_experiment._service = (mock_service := MagicMock())
    mock_service.get_job.return_value.status.side_effect = ["example_status"]

    statuses = abc_experiment.sample_statuses()
    assert statuses == ["example_status", None]

    mock_service.get_job.assert_called_once_with("example_job_id")


def test_clean_circuit(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment._samples = sample_circuits
    abc_experiment._clean_circuits()

    for sample in sample_circuits:
        assert sample.circuit[-1] == cirq.Moment(cirq.measure(*sample.circuit.all_qubits()))


def test_process_device_counts(abc_experiment: BenchmarkingExperiment) -> None:
    counts = {
        "00": 20,
        "01": 5,
        "11": 10,
    }
    probs = abc_experiment._process_device_counts(counts)

    assert probs == {"00": 20 / 35, "01": 5 / 35, "10": 0.0, "11": 10 / 35}


def test_retrieve_ss_jobs_not_all_submitted(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment._samples = sample_circuits
    abc_experiment._samples[0].job = "example_job_id"
    mock_job_1 = MagicMock()
    mock_job_1.status.return_value = "Queued"
    mock_job_1.job_id.return_value = "example_job_id"

    abc_experiment._service = MagicMock()
    abc_experiment._service.get_job.return_value = mock_job_1

    statuses = abc_experiment.retrieve_ss_jobs()

    abc_experiment._service.get_job.assert_called_once_with("example_job_id")

    assert statuses == {"example_job_id": "Queued"}
    assert not hasattr(sample_circuits[0], "probabilities")
    assert not hasattr(sample_circuits[1], "probabilities")


def test_retrieve_ss_jobs_nothing_to_retrieve(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment._samples = sample_circuits
    statuses = abc_experiment.retrieve_ss_jobs()
    assert statuses == {}


def test_retrieve_ss_jobs_all_submitted(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment._samples = sample_circuits
    abc_experiment._samples[0].job = "example_job_id_1"
    mock_job_1 = MagicMock()
    mock_job_1.status.return_value = "Queued"
    mock_job_1.job_id.return_value = "example_job_id_1"

    abc_experiment._samples[1].job = "example_job_id_2"
    mock_job_2 = MagicMock()
    mock_job_2.status.return_value = "Done"
    mock_job_2.job_id.return_value = "example_job_id_2"
    mock_job_2.counts.return_value = {"00": 5, "11": 10}

    abc_experiment._service = MagicMock()
    abc_experiment._service.get_job.side_effect = [mock_job_1, mock_job_2]

    statuses = abc_experiment.retrieve_ss_jobs()

    # Check get job calls
    abc_experiment._service.get_job.assert_has_calls(
        [call("example_job_id_1"), call("example_job_id_2")]
    )

    # Check counts call
    mock_job_2.counts.assert_called_once_with(0)

    assert statuses == {"example_job_id_1": "Queued"}

    # Check probabilities correctly updated
    assert sample_circuits[1].probabilities == {"00": 5 / 15, "01": 0.0, "10": 0.0, "11": 10 / 15}
    assert not hasattr(sample_circuits[0], "probabilities")


def test_collect_data_no_samples(abc_experiment: BenchmarkingExperiment) -> None:
    with pytest.raises(RuntimeError, match="The experiment has not yet ben run."):
        abc_experiment.collect_data()


def test_collect_data_no_jobs_to_retrieve(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    sample_circuits[0].probabilities = {"00": 1.0, "10": 0.0, "01": 0.0, "11": 0.0}
    sample_circuits[1].probabilities = {"00": 0.0, "10": 0.0, "01": 0.0, "11": 1.0}
    abc_experiment._samples = sample_circuits
    abc_experiment.process_probabilities = MagicMock()

    assert abc_experiment.collect_data()
    abc_experiment.process_probabilities.assert_called_once_with(sample_circuits)


def test_collect_data_no_jobs_to_retrieve_not_all_probabilities(
    abc_experiment: BenchmarkingExperiment,
    sample_circuits: list[Sample],
    capfd: pytest.CaptureFixture[str],
) -> None:
    sample_circuits[0].probabilities = {"00": 1.0, "10": 0.0, "01": 0.0, "11": 0.0}
    abc_experiment._samples = sample_circuits
    abc_experiment.process_probabilities = MagicMock()

    assert not abc_experiment.collect_data()
    out, _ = capfd.readouterr()
    assert out == "Some samples do not have probability results.\n"
    abc_experiment.process_probabilities.assert_not_called()


def test_collect_data_no_jobs_to_retrieve_not_all_probabilities_forced(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    sample_circuits[0].probabilities = {"00": 1.0, "10": 0.0, "01": 0.0, "11": 0.0}
    abc_experiment._samples = sample_circuits
    abc_experiment.process_probabilities = MagicMock()

    assert abc_experiment.collect_data(force=True)
    abc_experiment.process_probabilities.assert_called_once_with([sample_circuits[0]])


def test_collect_data_cannot_force(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment._samples = sample_circuits
    abc_experiment.process_probabilities = MagicMock()

    with pytest.raises(
        RuntimeError, match="Cannot force data collection when there are no completed samples."
    ):
        abc_experiment.collect_data(force=True)

    abc_experiment.process_probabilities.assert_not_called()


def test_collect_data_outstanding_jobs(
    abc_experiment: BenchmarkingExperiment,
    sample_circuits: list[Sample],
    capfd: pytest.CaptureFixture[str],
) -> None:
    abc_experiment._samples = sample_circuits
    abc_experiment.process_probabilities = MagicMock()
    abc_experiment.retrieve_ss_jobs = MagicMock(return_value={"example_id": "some_status"})
    assert not abc_experiment.collect_data()
    out, _ = capfd.readouterr()
    assert out == (
        "Not all circuits have been sampled. Please wait and try again.\n"
        "Outstanding Superstaq jobs:\n"
        "{'example_id': 'some_status'}\n"
    )
    abc_experiment.process_probabilities.assert_not_called()


def test_collect_data_outstanding_jobs_force(
    abc_experiment: BenchmarkingExperiment,
    sample_circuits: list[Sample],
    capfd: pytest.CaptureFixture[str],
) -> None:
    abc_experiment._samples = sample_circuits
    abc_experiment.process_probabilities = MagicMock(return_value=pd.DataFrame([{"data": 1.0}]))
    abc_experiment.samples[0].probabilities = {"00": 1.0, "10": 0.0, "01": 0.0, "11": 0.0}
    abc_experiment.retrieve_ss_jobs = MagicMock(return_value={"example_id": "some_status"})
    assert abc_experiment.collect_data(force=True)
    out, _ = capfd.readouterr()
    assert out == (
        "Not all circuits have been sampled. Please wait and try again.\n"
        "Outstanding Superstaq jobs:\n"
        "{'example_id': 'some_status'}\n"
        "Some samples do not have probability results.\n"
    )

    abc_experiment.process_probabilities.assert_called_once_with([abc_experiment.samples[0]])

    pd.testing.assert_frame_equal(pd.DataFrame([{"data": 1.0}]), abc_experiment.raw_data)
