# Copyright 2025 Infleqtion
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
import cirq
import numpy as np

from cirq_superstaq import msd_5_to_1, msd_7_to_1, msd_15_to_1


def test_5_to_1_msd() -> None:
    qubits = cirq.LineQubit.range(5)
    sim = cirq.Simulator()
    magic_state_circuit = msd_5_to_1(qubits)
    sim_results = sim.simulate(magic_state_circuit)
    state_vector = sim_results.final_state_vector

    density_matrix = cirq.density_matrix_from_state_vector(state_vector, indices=[0])
    # The expected density matrix below corresponds to the output of the 5-to-1 magic
    # state distillation protocol.
    # These values are derived from the theoretical output state after distillation,
    # which is a specific magic state
    # (see e.g. Bravyi & Kitaev, Phys. Rev. A 71, 022316 (2005)). The entries represent
    # the expected density matrix
    # for the distilled qubit in the computational basis.
    expected_density = np.array(
        [
            [0.7886752 + 0.00000000j, 0.28867516 + -0.2886752j],
            [0.28867516 + 0.2886752j, 0.2113249 + 0.00000000j],
        ],
        dtype=np.complex64,
    )
    assert np.allclose(density_matrix, expected_density, rtol=1e-05)


def test_7_to_1_msd() -> None:
    qubits = cirq.LineQubit.range(8)
    sim = cirq.Simulator()
    magic_state_circuit = msd_7_to_1(qubits)
    sim_results = sim.simulate(magic_state_circuit)
    state_vector = sim_results.final_state_vector
    magic_state = sim_results.get_state_containing_qubit(cirq.q(0)).target_tensor

    assert (cirq.dirac_notation(magic_state)) == "0.71|0⟩ + 0.71j|1⟩"

    density_matrix = cirq.density_matrix_from_state_vector(state_vector, indices=[0])
    expected_density = np.array(
        [
            [0.50000006 + 0.0j, 0.0 - 0.50000006j],
            [0.0 + 0.50000006j, 0.50000006 + 0.0j],
        ],
        dtype=np.complex64,
    )
    assert np.allclose(density_matrix, expected_density, rtol=1e-05)


def test_15_to_1_msd() -> None:
    qubits = cirq.LineQubit.range(16)
    sim = cirq.Simulator()
    magic_state_circuit = msd_15_to_1(qubits)
    sim_results = sim.simulate(magic_state_circuit)
    state_vector = sim_results.final_state_vector
    magic_state = sim_results.get_state_containing_qubit(qubits[15]).target_tensor

    assert (cirq.dirac_notation(magic_state)) == "0.71|0⟩ + (0.5+0.5j)|1⟩"

    density_matrix = cirq.density_matrix_from_state_vector(state_vector, indices=[15])
    expected_density = np.array(
        [
            [0.49999997 + 0.0j, 0.35355338 - 0.35355338j],
            [0.35355338 + 0.35355338j, 0.5 + 0.0j],
        ],
        dtype=np.complex64,
    )
    assert np.allclose(density_matrix, expected_density, rtol=1e-05)
