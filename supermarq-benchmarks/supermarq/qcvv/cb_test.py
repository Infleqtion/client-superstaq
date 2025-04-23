# Copyright 2021 The Cirq Developers
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

import itertools
import os
from unittest.mock import MagicMock, patch

import cirq
import cirq.testing
import numpy as np
import pandas as pd
import pytest

from supermarq.qcvv import CB, Sample


@pytest.fixture(scope="session", autouse=True)
def patch_tqdm() -> None:
    os.environ["TQDM_DISABLE"] = "1"


def test_cb_init() -> None:
    with patch("cirq_superstaq.service.Service"):

        with pytest.raises(
            RuntimeError, 
            match="This cycle benchmarking is only valid for Clifford elements."
        ):  
            qubit = cirq.LineQubit(0)
            circuit = cirq.Circuit([cirq.T(qubit)])
            CB(circuit, pauli_channels=1)

        with pytest.raises(
            RuntimeError, 
            match="All Pauli channels must be over 1 qubits. XX is over 2 qubits."
        ):  
            qubit = cirq.LineQubit(0)
            circuit = cirq.Circuit([cirq.X(qubit)])
            CB(circuit, pauli_channels=["XX"])

        with pytest.raises(
            RuntimeError, 
            match="All Pauli channels must be over 2 qubits. Y is over 1 qubits."
        ):  
            qubits = cirq.LineQubit.range(2)
            circuit = cirq.Circuit([cirq.X(qubits[0]), cirq.Z(qubits[1])])
            CB(circuit, pauli_channels=["Y"])
        
        qubit = cirq.LineQubit(0)
        circuit = cirq.Circuit([cirq.X(qubit), cirq.H(qubit)])

        experiment = CB(circuit, pauli_channels=3)
        assert experiment.num_qubits == 1
        assert len(experiment.pauli_channels) == 3
        assert experiment._dressed_measurement
        assert experiment._matrix_order == 4
        assert experiment.cycle_depths == [4, 8]

        experiment = CB(circuit, pauli_channels=6)
        assert len(experiment.pauli_channels) == 4

        experiment = CB(circuit, pauli_channels=6, dressed_measurement=False)
        assert not experiment._dressed_measurement

        experiment = CB(circuit, pauli_channels=["X", "Z"])
        assert len(experiment.pauli_channels) == 2

        experiment = CB(circuit, pauli_channels=["X", "Z", "Z"])
        assert len(experiment.pauli_channels) == 2

        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
                                    cirq.H(qubits[0]), 
                                    cirq.CX(qubits[0], qubits[1]),
                                    cirq.H(qubits[0])
                                ])

        experiment = CB(circuit, pauli_channels=5)
        assert experiment.num_qubits == 2
        assert experiment._dressed_measurement
        assert experiment._matrix_order == 2
        assert len(experiment.pauli_channels) == 5

        experiment = CB(circuit, pauli_channels=20)
        assert len(experiment.pauli_channels) == 16
        

@pytest.fixture
def cb_experiment() -> CB:
    with patch("cirq_superstaq.service.Service"):
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
                                    cirq.H(qubits[0]), 
                                    cirq.CX(qubits[0], qubits[1]),
                                    cirq.H(qubits[0])
                                ])
        pauli_channels = ["XY", "YZ"]
        return CB(circuit, pauli_channels=pauli_channels)


# def test_build_circuits(cb_experiment: CB) -> None:
#     with patch("supermarq.qcvv.xeb.random.choices") as random_choices:
#         random_choices.side_effect = [
#             [p1, p2] for p1, p2 in itertools.product("IXYZ", "IXYZ")
#         ]
#         samples = cb_experiment._build_circuits(1, [1, 2])

#     assert len(samples) == 2
#     qubits = cb_experiment.qubits
#     cirq.testing.assert_same_circuits(
#         samples[0].raw_circuit,
#         cirq.Circuit(
#             [
#                 cirq.Y(qubits[0])**0.5,
#                 cirq.X(qubits[1])**(-0.5),
#                 cirq.I(qubits[0]),
#                 cirq.I(qubits[1]),
#                 cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
#                 cirq.TaggedOperation(cirq.CX(qubits[0], qubits[1]), "no_compile"),
#                 cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
#                 cirq.I(qubits[0]),
#                 cirq.X(qubits[1]),
#                 cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
#                 cirq.TaggedOperation(cirq.CX(qubits[0], qubits[1]), "no_compile"),
#                 cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
#                 cirq.I(qubits[0]),
#                 cirq.Y(qubits[1]),
#                 cirq.Y(qubits[0])**(-0.5),
#                 cirq.X(qubits[1])**0.5,
#                 cirq.measure(qubits)
#             ]
#         )
#     )
#     assert samples[0].data == {"circuit_depth": 5, "num_cycles": 2, "two_qubit_gate": "CZ"}
 