import cirq
import numpy
from MSD_15_to_1 import msd_15_to_1


def test_15_to_1_msd() -> None:
    qubits = cirq.LineQubit.range(16)
    sim = cirq.Simulator()
    magic_state_circuit = msd_15_to_1(qubits)
    sim_results = sim.simulate(magic_state_circuit)
    state_vector = sim_results.final_state_vector
    magic_state = sim_results.get_state_containing_qubit(cirq.q(15)).target_tensor

    assert (cirq.dirac_notation(magic_state)) == "0.71|0⟩ + (0.5+0.5j)|1⟩"

    density_matrix = cirq.density_matrix_from_state_vector(state_vector, indices=[15])
    tomo_res = cirq.experiments.single_qubit_state_tomography(
        sim, qubits[15], magic_state_circuit, 1000
    )

    expected_density = numpy.array(
        [
            [0.49999997 + 0.0j, 0.35355338 - 0.35355338j],
            [0.35355338 + 0.35355338j, 0.5 + 0.0j],
        ],
        dtype=numpy.complex64,
    )
    assert numpy.allclose(density_matrix, expected_density, rtol=1e-05)
    assert numpy.allclose(expected_density, tomo_res.data, rtol=0.1)
    assert numpy.allclose(density_matrix, tomo_res.data, rtol=0.1)
