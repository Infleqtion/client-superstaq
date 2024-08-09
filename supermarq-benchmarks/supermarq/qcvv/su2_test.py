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
import textwrap
from unittest.mock import MagicMock, patch

import cirq
import pandas as pd
import pytest

from supermarq.qcvv import SU2, Sample


@pytest.fixture(scope="session", autouse=True)
def patch_tqdm() -> None:
    os.environ["TQDM_DISABLE"] = "1"


def test_su2_init() -> None:
    with patch("cirq_superstaq.service.Service"):
        experiment = SU2(cirq.CNOT)
    assert experiment.num_qubits == 2
    assert experiment.two_qubit_gate == cirq.CNOT


def test_su2_init_raises() -> None:
    with patch("cirq_superstaq.service.Service"):
        with pytest.raises(
            ValueError,
            match="The `two_qubit_gate` parameter must be a gate that acts on exactly two qubits.",
        ):
            SU2(cirq.X)


@pytest.fixture
def su2_experiment() -> SU2:
    with patch("cirq_superstaq.service.Service"):
        return SU2(cirq.CNOT)


def test_build_circuits(su2_experiment: SU2) -> None:
    # By extension also tests the _component() method
    # Patch the PhasedXZ drawing function to ignore the rotation parameters
    cirq.PhasedXZGate._circuit_diagram_info_ = MagicMock(
        return_value=cirq.CircuitDiagramInfo(wire_symbols=["PhXZ"])
    )

    samples = su2_experiment._build_circuits(2, [1, 3])
    assert len(samples) == 4
    for sample in samples:
        assert "num_two_qubit_gates" in sample.data
        assert sample.data["num_two_qubit_gates"] in [2, 6]
        if sample.data["num_two_qubit_gates"] == 2:
            cirq.testing.assert_has_diagram(
                sample.circuit,
                textwrap.dedent(
                    """
                    0: ───PhXZ───@───X───@───PhXZ───X───PhXZ───X───PhXZ───M───
                                 │       │                                │
                    1: ───PhXZ───X───X───X───PhXZ───X───PhXZ───X───PhXZ───M───
                    """
                ),
            )
        else:
            cirq.testing.assert_has_diagram(
                sample.circuit,
                textwrap.dedent(
                    """
                    0: ───PhXZ───@───X───@───PhXZ───@───X───@───PhXZ───@───X───@───PhXZ───M───
                                 │       │          │       │          │       │          │
                    1: ───PhXZ───X───X───X───PhXZ───X───X───X───PhXZ───X───X───X───PhXZ───M───
                    """
                ),
            )


def test_process_probabilities(su2_experiment: SU2) -> None:
    samples = [
        Sample(
            circuit=MagicMock(),
            data={"num_two_qubit_gates": 2},
        ),
        Sample(
            circuit=MagicMock(),
            data={"num_two_qubit_gates": 4},
        ),
        Sample(
            circuit=MagicMock(),
            data={"num_two_qubit_gates": 6},
        ),
    ]
    samples[0].probabilities = {"00": 0.1, "01": 0.2, "10": 0.3, "11": 0.4}
    samples[1].probabilities = {"00": 0.1, "01": 0.4, "10": 0.3, "11": 0.2}
    samples[2].probabilities = {"00": 0.3, "01": 0.2, "10": 0.1, "11": 0.4}

    raw_data = su2_experiment._process_probabilities(samples)
    expected_df = pd.DataFrame(
        [
            {"num_two_qubit_gates": 2, "00": 0.1, "01": 0.2, "10": 0.3, "11": 0.4},
            {"num_two_qubit_gates": 4, "00": 0.1, "01": 0.4, "10": 0.3, "11": 0.2},
            {"num_two_qubit_gates": 6, "00": 0.3, "01": 0.2, "10": 0.1, "11": 0.4},
        ]
    )
    pd.testing.assert_frame_equal(raw_data, expected_df)


def test_analyse_results(su2_experiment: SU2) -> None:
    su2_experiment._samples = [
        Sample(
            circuit=MagicMock(),
            data={"num_two_qubit_gates": 2},
        ),
        Sample(
            circuit=MagicMock(),
            data={"num_two_qubit_gates": 4},
        ),
        Sample(
            circuit=MagicMock(),
            data={"num_two_qubit_gates": 6},
        ),
        Sample(
            circuit=MagicMock(),
            data={"num_two_qubit_gates": 8},
        ),
    ]

    def decay(x):  # type: ignore[no-untyped-def] # pylint: disable=W9012
        return (3 * 0.75 * 0.975**x + 1) / 4

    # Generate decay data from a known formula
    su2_experiment._raw_data = pd.DataFrame(
        [
            {"num_two_qubit_gates": 2, "00": decay(2), "01": 0.0, "10": 0.0, "11": 1 - decay(2)},
            {"num_two_qubit_gates": 4, "00": decay(4), "01": 0.0, "10": 0.0, "11": 1 - decay(4)},
            {"num_two_qubit_gates": 6, "00": decay(6), "01": 0.0, "10": 0.0, "11": 1 - decay(6)},
            {"num_two_qubit_gates": 8, "00": decay(8), "01": 0.0, "10": 0.0, "11": 1 - decay(8)},
        ]
    )
    result = su2_experiment.analyse_results(plot_results=True)

    assert result.two_qubit_gate_fidelity == pytest.approx(0.975)
    assert result.two_qubit_gate_fidelity_std == pytest.approx(0.0)
    assert result.two_qubit_gate_error == pytest.approx(0.025)
    assert result.two_qubit_gate_error_std == pytest.approx(0.0)
    assert result.single_qubit_noise == pytest.approx(0.25)
    assert result.single_qubit_noise_std == pytest.approx(0.0)


def test_haar_random_rotation() -> None:
    r = SU2._haar_random_rotation()
    assert isinstance(r, cirq.Gate)
    assert r.num_qubits() == 1
