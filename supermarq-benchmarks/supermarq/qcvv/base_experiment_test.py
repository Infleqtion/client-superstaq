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
from general_superstaq.superstaq_exceptions import SuperstaqException

from supermarq.qcvv.base_experiment import BenchmarkingExperiment, Sample


@pytest.fixture(scope="session", autouse=True)
def patch_tqdm() -> None:
    os.environ["TQDM_DISABLE"] = "1"


@pytest.fixture
@patch.multiple(BenchmarkingExperiment, __abstractmethods__=set())
def abc_experiment() -> BenchmarkingExperiment:
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
    abc_experiment._results = pd.DataFrame([{"example": 1234}])
    abc_experiment.build_circuits = MagicMock()
    abc_experiment.run_ss_jobs = MagicMock()
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
    abc_experiment.run_ss_jobs = (mock_run_ss_jobs := MagicMock())
    abc_experiment.sample_circuits_with_simulator = (
        mock_sample_circuits_with_simulator := MagicMock()
    )
    abc_experiment.process_probabilities = (mock_process_probabilities := MagicMock())

    abc_experiment.run(50, [1, 50, 100], shots=50)

    mock_build_circuits.assert_called_once_with(50, [1, 50, 100])

    mock_sample_circuits_with_simulator.assert_called_once()
    call_args = mock_sample_circuits_with_simulator.call_args_list[0][0]
    assert call_args[1] == 50
    assert isinstance(call_args[0], cirq.Simulator)  # Test simulated on a default target

    mock_process_probabilities.assert_called_once_with()
    mock_run_ss_jobs.assert_not_called()


def test_run_local_defined_sim(abc_experiment: BenchmarkingExperiment) -> None:
    abc_experiment.build_circuits = (mock_build_circuits := MagicMock())
    abc_experiment.run_ss_jobs = (mock_run_ss_jobs := MagicMock())
    abc_experiment.sample_circuits_with_simulator = (
        mock_sample_circuits_with_simulator := MagicMock()
    )
    abc_experiment.process_probabilities = (mock_process_probabilities := MagicMock())

    abc_experiment.run(
        50, [1, 50, 100], shots=50, target=(target_sim := cirq.DensityMatrixSimulator())
    )

    mock_build_circuits.assert_called_once_with(50, [1, 50, 100])

    mock_sample_circuits_with_simulator.assert_called_once()
    call_args = mock_sample_circuits_with_simulator.call_args_list[0][0]
    assert call_args[1] == 50
    assert call_args[0] == target_sim  # Test simulated on the given target

    mock_process_probabilities.assert_called_once_with()
    mock_run_ss_jobs.assert_not_called()


def test_run_on_ss_server(abc_experiment: BenchmarkingExperiment) -> None:
    abc_experiment.build_circuits = (mock_build_circuits := MagicMock())
    abc_experiment.run_ss_jobs = (mock_run_ss_jobs := MagicMock())
    abc_experiment.sample_circuits_with_simulator = (
        mock_sample_circuits_with_simulator := MagicMock()
    )
    abc_experiment.process_probabilities = (mock_process_probabilities := MagicMock())

    abc_experiment.run(50, [1, 50, 100], shots=50, target="example_ss_target")

    mock_build_circuits.assert_called_once_with(50, [1, 50, 100])

    mock_sample_circuits_with_simulator.assert_not_called()

    mock_process_probabilities.assert_called_once_with()
    mock_run_ss_jobs.assert_called_once_with("example_ss_target", 50, False)


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


def test_run_ss_jobs(abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]) -> None:

    abc_experiment._samples = sample_circuits
    with patch("cirq_superstaq.qcvv.base_experiment.Service") as mock_service:
        mock_service().create_job.return_value = MagicMock(
            counts=MagicMock(
                return_value=[
                    {"00": 5, "01": 5, "10": 5, "11": 10},
                    {"00": 5, "01": 5, "10": 5, "11": 10},
                ]
            )
        )
        mock_service().target_info.return_value = {}
        abc_experiment.run_ss_jobs("example_target", shots=100)

    for sample in sample_circuits:
        assert sample.probabilities == {"00": 0.2, "01": 0.2, "10": 0.2, "11": 0.4}

    mock_service().create_job.assert_called_once_with(
        [sample.circuit for sample in sample_circuits],
        target="example_target",
        method=None,
        repetitions=100,
    )


def test_run_ss_jobs_not_all_samples(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:

    abc_experiment._samples = sample_circuits
    abc_experiment._samples[0].probabilities = {"example": 0.7}
    with patch("cirq_superstaq.qcvv.base_experiment.Service") as mock_service:
        mock_service().create_job.return_value = MagicMock(
            counts=MagicMock(
                return_value=[
                    {"00": 5, "01": 5, "10": 5, "11": 10},
                    {"00": 5, "01": 5, "10": 5, "11": 10},
                ]
            )
        )
        mock_service().target_info.return_value = {}
        abc_experiment.run_ss_jobs("example_target", shots=100, all_samples=False)

    assert sample_circuits[1].probabilities == {"00": 0.2, "01": 0.2, "10": 0.2, "11": 0.4}

    mock_service().create_job.assert_called_once_with(
        [sample_circuits[1].circuit],
        target="example_target",
        method=None,
        repetitions=100,
    )


def test_run_ss_jobs_dry_run_partitioning(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:

    abc_experiment._samples = sample_circuits
    with patch("cirq_superstaq.qcvv.base_experiment.Service") as mock_service:
        mock_service().create_job.return_value = MagicMock(
            counts=MagicMock(
                return_value=[
                    {"00": 5, "01": 5, "10": 5, "11": 10},
                    {"00": 5, "01": 5, "10": 5, "11": 10},
                ]
            )
        )
        mock_service().target_info.return_value = {"max_experiments": 1}
        abc_experiment.run_ss_jobs("example_target", shots=100, dry_run=True)

    for sample in sample_circuits:
        assert sample.probabilities == {"00": 0.2, "01": 0.2, "10": 0.2, "11": 0.4}

    mock_service().create_job.assert_has_calls(
        [
            call(
                [sample_circuits[0].circuit],
                target="example_target",
                method="dry-run",
                repetitions=100,
            ),
            call().counts(),
            call(
                [sample_circuits[1].circuit],
                target="example_target",
                method="dry-run",
                repetitions=100,
            ),
            call().counts(),
        ]
    )


def test_run_ss_jobs_with_exception(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment._samples = sample_circuits
    with patch("cirq_superstaq.qcvv.base_experiment.Service") as mock_service:
        mock_service().create_job.return_value = MagicMock(
            counts=MagicMock(side_effect=SuperstaqException("example_exception"))
        )
        with pytest.warns(UserWarning):
            abc_experiment.run_ss_jobs("example_target", shots=100, dry_run=True)


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


def test_clean_circuit(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment._samples = sample_circuits
    abc_experiment._clean_circuits()

    for sample in sample_circuits:
        assert sample.circuit[-1] == cirq.Moment(cirq.measure(*sample.circuit.all_qubits()))


def test_process_probabilities(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment._samples = sample_circuits
    with pytest.raises(RuntimeError, match="Not all circuits have been successfully sampled."):
        abc_experiment.process_probabilities()


def test_run_ss_jobs_counts_not_list(
    abc_experiment: BenchmarkingExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment._samples = sample_circuits
    with patch("cirq_superstaq.qcvv.base_experiment.Service") as mock_service:
        mock_service().create_job.return_value = MagicMock(
            counts=MagicMock(return_value={"00": 5, "01": 5, "10": 5, "11": 10})
        )
        mock_service().target_info.return_value = {}
        with pytest.raises(
            TypeError, match="Expected the counts returned from the `job` to be a list."
        ):
            abc_experiment.run_ss_jobs("example_target", shots=100)
