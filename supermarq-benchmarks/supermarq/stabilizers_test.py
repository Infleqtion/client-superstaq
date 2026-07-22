# Copyright 2026 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# limitations under the License.

from __future__ import annotations

from unittest.mock import patch

import numpy as np

import supermarq
from supermarq import stabilizers
from supermarq.benchmarks.mermin_bell import MerminBell


def test_construct_stabilizer() -> None:
    mb = MerminBell(3)
    mermin_op = MerminBell._mermin_operator(mb, num_qubits=3)
    assert mb.score(supermarq.simulation.get_ideal_counts(mb.circuit())) == 1

    mb = MerminBell(5)
    mermin_op = MerminBell._mermin_operator(mb, num_qubits=3)
    stabilizers.construct_stabilizer(num_qubits=3, clique=[(0.25, mermin_op)])


def test_prepare_x_matrix() -> None:
    mb = MerminBell(5)
    measurement_circuit = MerminBell._get_measurement_circuit(mb)
    stabilizers.prepare_x_matrix(measurement_circuit)
    N = measurement_circuit.num_qubits
    with patch("numpy.linalg.matrix_rank", return_value=N - 1):
        stabilizers.prepare_x_matrix(measurement_circuit)


def test_patch_z_matrix() -> None:
    mb = MerminBell(5)
    measurement_circuit = MerminBell._get_measurement_circuit(mb)
    stabilizers.prepare_x_matrix(measurement_circuit)
    N = measurement_circuit.num_qubits

    assert stabilizers.patch_z_matrix(measurement_circuit) is None

    with patch("supermarq.stabilizers.MeasurementCircuit.get_stabilizer", return_value=np.eye(N)):
        assert stabilizers.patch_z_matrix(measurement_circuit) is None
