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
# pylint: disable=missing-return-doc
# mypy: disable-error-code=method-assign

from __future__ import annotations

import os
from dataclasses import dataclass
from unittest.mock import MagicMock, call, patch

import cirq
import cirq_superstaq as css
import numpy as np
import pandas as pd
import pytest
from general_superstaq.superstaq_exceptions import SuperstaqServerException

from supermarq.qcvv.base_experiment import BenchmarkingExperiment, QCVVResults, Sample


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
            circuit=cirq.Circuit(cirq.CZ(*qubits), cirq.CZ(*qubits), cirq.measure(*qubits)),
            data={"circuit": 1},
        ),
        Sample(circuit=cirq.Circuit(cirq.CX(*qubits)), data={"circuit": 2}),
    ]


@dataclass(kw_only=True, frozen=True)
class ExampleResults(QCVVResults):
    """NamedTuple instance to use for testing"""

    example: float


def test_sample_target_property() -> None:
    sample = Sample(circuit=MagicMock(), data={})
    assert sample.target == "No target"

    sample.probabilities = {"0": 0.25, "1": 0.75}
    assert sample.target == "Local simulator"

    sample.job = MagicMock()
    sample.job.target.return_value = "Example target"
    assert sample.target == "Example target"


def test_benchmarking_experiment_init(abc_experiment: BenchmarkingExperiment) -> None:
    assert abc_experiment.num_qubits == 2
    assert abc_experiment._raw_data is None
    assert abc_experiment._results is None
    assert abc_experiment._samples is None

    abc_experiment._raw_data = pd.DataFrame([{"Example": 0.1}])
    abc_experiment._results = ExampleResults(
        experiment_name="Example", target="Some target", total_circuits=2, example=5.0
    )

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


def test_prepare_experiment_overwrite_error(abc_experiment: BenchmarkingExperiment) -> None:
    abc_experiment._samples = [Sample(circuit=MagicMock(), data={})]
    abc_experiment.build_circuits = MagicMock()

    with pytest.raises(
        RuntimeError,
        match="This experiment already has existing data which would be overwritten by "
        "rerunning the experiment. If this is the desired behavior set `overwrite=True`",
    ):
        abc_experiment._prepare_experiment(100, [1, 50, 100])


def test_prepare_experiment_overwrite(abc_experiment: BenchmarkingExperiment) -> None:
    abc_experiment._samples = [Sample(circuit=MagicMock(), data={})]
    abc_experiment.build_circuits = MagicMock()
    abc_experiment._validate_circuits = MagicMock()

    abc_experiment._prepare_experiment(100, [1, 50, 100], overwrite=True)

    abc_experiment.build_circuits.assert_called_once_with(100, [1, 50, 100])
    abc_experiment._validate_circuits.assert_called_once_with()


def test_prepare_experiment_with_bad_layers(abc_experiment: BenchmarkingExperiment) -> None:
    with pytest.raises(
        ValueError, match="The `cycle_depths` iterator can only include positive values."
    ):
        abc_experiment._prepare_experiment(20, [0])


def test_run_with_simulator(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    cirq.measurement_key_name = MagicMock()
    abc_experiment._prepare_experiment = MagicMock()
    abc_experiment._samples = sample_circuits
    test_target = MagicMock()
    mock_result = MagicMock()
    mock_result.histogram.return_value = {0: 0, 1: 100, 2: 0, 3: 0}
    test_target.run.return_value = mock_result

    abc_experiment.run_with_simulator(10, [5, 10, 20], target=test_target, shots=100)

    # Test simulator calls
    test_target.run.assert_has_calls(
        [
            call(sample_circuits[0].circuit, repetitions=100),
            call(sample_circuits[1].circuit, repetitions=100),
        ],
        any_order=True,
    )

    # Test probabilities
    assert sample_circuits[0].probabilities == {"00": 0.0, "01": 1.0, "10": 0.0, "11": 0.0}
    assert sample_circuits[1].probabilities == {"00": 0.0, "01": 1.0, "10": 0.0, "11": 0.0}

    abc_experiment._prepare_experiment.assert_called_once_with(10, [5, 10, 20], False)


def test_run_with_simulator_default_target(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    cirq.measurement_key_name = MagicMock()
    cirq.Simulator = (target := MagicMock())  # type: ignore [misc]
    abc_experiment._prepare_experiment = MagicMock()
    abc_experiment._samples = sample_circuits
    mock_result = MagicMock()
    mock_result.histogram.return_value = {0: 0, 1: 100, 2: 0, 3: 0}
    target().run.return_value = mock_result

    abc_experiment.run_with_simulator(10, [5, 10, 20], shots=100)

    # Test simulator calls
    target().run.assert_has_calls(
        [
            call(sample_circuits[0].circuit, repetitions=100),
            call(sample_circuits[1].circuit, repetitions=100),
        ],
        any_order=True,
    )

    # Test probabilities
    assert sample_circuits[0].probabilities == {"00": 0.0, "01": 1.0, "10": 0.0, "11": 0.0}
    assert sample_circuits[1].probabilities == {"00": 0.0, "01": 1.0, "10": 0.0, "11": 0.0}

    abc_experiment._prepare_experiment.assert_called_once_with(10, [5, 10, 20], False)


def test_run_on_device(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment._prepare_experiment = MagicMock()
    abc_experiment._samples = sample_circuits
    abc_experiment._service = (mock_service := MagicMock())
    mock_service().create_job.side_effect = [MagicMock(job_id="job_1"), MagicMock(job_id="job_2")]

    mock_service().target_info.return_value = {}
    abc_experiment.run_on_device(
        10, [5, 10, 20], target="example_target", shots=100, overwrite=False, **{"some": "options"}
    )

    mock_service.create_job.assert_has_calls(
        [
            call(
                sample_circuits[0].circuit,
                target="example_target",
                method=None,
                repetitions=100,
                some="options",
            ),
            call(
                sample_circuits[1].circuit,
                target="example_target",
                method=None,
                repetitions=100,
                some="options",
            ),
        ],
        any_order=True,
    )

    abc_experiment._prepare_experiment.assert_called_once_with(10, [5, 10, 20], False)


def test_run_on_device_dry_run(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment._prepare_experiment = MagicMock()
    abc_experiment._samples = sample_circuits
    abc_experiment._service = (mock_service := MagicMock())
    mock_service().create_job.side_effect = [MagicMock(job_id="job_1"), MagicMock(job_id="job_2")]

    mock_service().target_info.return_value = {}
    abc_experiment.run_on_device(
        10, [5, 10, 20], target="example_target", shots=100, method="dry-run"
    )

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


def test_run_on_device_job_already_has_id(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment._prepare_experiment = MagicMock()
    abc_experiment._samples = sample_circuits
    sample_circuits[0].job = MagicMock()
    abc_experiment._service = (mock_service := MagicMock())
    mock_service().create_job.side_effect = [MagicMock(job_id="job_1"), MagicMock(job_id="job_2")]

    mock_service().target_info.return_value = {}
    abc_experiment.run_on_device(
        10, [5, 10, 20], target="example_target", shots=100, method="example_method"
    )

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


def test_run_on_device_with_exception(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment._samples = sample_circuits
    abc_experiment._service = (mock_service := MagicMock())
    abc_experiment._prepare_experiment = MagicMock()

    mock_service.create_job.side_effect = SuperstaqServerException("example_exception")
    with pytest.warns(UserWarning):
        abc_experiment.run_on_device(10, [5, 10, 20], target="example_target", shots=100)


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
    mock_job = MagicMock(spec=css.Job)
    mock_job.status.return_value = "example_status"
    sample_circuits[0].job = mock_job

    statuses = abc_experiment.sample_statuses()
    assert statuses == ["example_status", None]
    mock_job.status.assert_called_once_with()


def test_sample_targets(
    abc_experiment: BenchmarkingExperiment,
) -> None:
    abc_experiment._samples = [mock_sample_0 := MagicMock(), mock_sample_1 := MagicMock()]
    mock_sample_0.target = "target_0"
    mock_sample_1.target = "target_1"
    assert abc_experiment.sample_targets == ["target_0", "target_1"]

    mock_sample_1.target = "target_0"
    assert abc_experiment.sample_targets == ["target_0"]


def test_validate_circuits(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment._samples = sample_circuits
    # Should't get any errors with the base circuits
    abc_experiment._validate_circuits()

    # Add a gate so not all measurements are terminal
    abc_experiment._samples[0].circuit += cirq.X(abc_experiment.qubits[0])
    with pytest.raises(
        ValueError, match="QCVV experiment circuits can only contain terminal measurements"
    ):
        abc_experiment._validate_circuits()

    # Remove measurements
    abc_experiment._samples[0].circuit = abc_experiment._samples[0].circuit[:-2] + cirq.measure(
        abc_experiment.qubits[0]
    )
    with pytest.raises(
        ValueError,
        match="The terminal measurement in QCVV experiment circuits must measure all qubits.",
    ):
        abc_experiment._validate_circuits()


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

    mock_job_1 = MagicMock()
    mock_job_1.status.return_value = "Queued"
    mock_job_1.job_id.return_value = "example_job_id"
    abc_experiment._samples[0].job = mock_job_1

    statuses = abc_experiment.retrieve_ss_jobs()

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
    mock_job_1 = MagicMock(spec=css.Job)
    mock_job_1.status.return_value = "Queued"
    mock_job_1.job_id.return_value = "example_job_id_1"
    abc_experiment._samples[0].job = mock_job_1

    mock_job_2 = MagicMock(spec=css.Job)
    mock_job_2.status.return_value = "Done"
    mock_job_2.job_id.return_value = "example_job_id_2"
    mock_job_2.counts.return_value = {"00": 5, "11": 10}
    abc_experiment._samples[1].job = mock_job_2

    statuses = abc_experiment.retrieve_ss_jobs()

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
