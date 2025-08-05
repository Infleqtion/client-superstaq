import cirq
import numpy

from .msd import msd_5_to_1, msd_7_to_1, msd_15_to_1


def test_5_to_1_msd() -> None:
    qubits = cirq.LineQubit.range(5)
    sim = cirq.Simulator()
    magic_state_circuit = msd_5_to_1(qubits)
    sim_results = sim.simulate(magic_state_circuit)
    state_vector = sim_results.final_state_vector
    magic_state = sim_results.get_state_containing_qubit(cirq.q(0)).target_tensor

    assert (cirq.dirac_notation(magic_state)) == "0.89|0⟩ + (0.33+0.33j)|1⟩"

    density_matrix = cirq.density_matrix_from_state_vector(state_vector, indices=[0])
    expected_density = numpy.array(
        [
            [0.7886752 + 0.00000000j, 0.28867516 + -0.2886752j],
            [0.28867516 + 0.2886752j, 0.2113249 + 0.00000000j],
        ],
        dtype=numpy.complex64,
    )
    assert numpy.allclose(density_matrix, expected_density, rtol=1e-05)


def test_7_to_1_msd() -> None:
    qubits = cirq.LineQubit.range(8)
    sim = cirq.Simulator()
    magic_state_circuit = msd_7_to_1(qubits)
    sim_results = sim.simulate(magic_state_circuit)
    state_vector = sim_results.final_state_vector
    magic_state = sim_results.get_state_containing_qubit(cirq.q(0)).target_tensor

    assert (cirq.dirac_notation(magic_state)) == "0.71|0⟩ + 0.71j|1⟩"

    density_matrix = cirq.density_matrix_from_state_vector(state_vector, indices=[0])
    expected_density = numpy.array(
        [
            [0.50000006 + 0.0j, 0.0 - 0.50000006j],
            [0.0 + 0.50000006j, 0.50000006 + 0.0j],
        ],
        dtype=numpy.complex64,
    )
    assert numpy.allclose(density_matrix, expected_density, rtol=1e-05)


def test_15_to_1_msd() -> None:
    qubits = cirq.LineQubit.range(16)
    sim = cirq.Simulator()
    magic_state_circuit = msd_15_to_1(qubits)
    sim_results = sim.simulate(magic_state_circuit)
    state_vector = sim_results.final_state_vector
    magic_state = sim_results.get_state_containing_qubit(cirq.q(15)).target_tensor

    assert (cirq.dirac_notation(magic_state)) == "0.71|0⟩ + (0.5+0.5j)|1⟩"

    density_matrix = cirq.density_matrix_from_state_vector(state_vector, indices=[15])
    expected_density = numpy.array(
        [
            [0.49999997 + 0.0j, 0.35355338 - 0.35355338j],
            [0.35355338 + 0.35355338j, 0.5 + 0.0j],
        ],
        dtype=numpy.complex64,
    )
    assert numpy.allclose(density_matrix, expected_density, rtol=1e-05)
