import cirq
import numpy
from MSD_7_to_1 import msd_7_to_1


def test_consistent_state() -> None:
    qubits = cirq.LineQubit.range(8)
    sim = cirq.Simulator()
    magicStateCir = msd_7_to_1(qubits)

    fMap: dict[str, int] = {}
    for _i in range(1000):
        targetTensorResults = cirq.dirac_notation(
            sim.simulate(magicStateCir).get_state_containing_qubit(cirq.q(0)).target_tensor
        )
        if targetTensorResults in fMap:
            fMap[targetTensorResults] += 1
        else:
            fMap[targetTensorResults] = 1

    assert len(fMap) == 1
    assert str(fMap) == "{'0.71|0⟩ + 0.71j|1⟩': 1000}"


def test_similarities() -> None:
    qubits = cirq.LineQubit.range(8)
    sim = cirq.Simulator()
    magicStateCir = msd_7_to_1(qubits)
    stateVector = sim.simulate(magicStateCir).final_state_vector

    density_matrix = cirq.density_matrix_from_state_vector(stateVector, indices=[0])
    tomo_res = cirq.experiments.single_qubit_state_tomography(sim, qubits[0], magicStateCir, 1000)

    expected_density = numpy.array(
        [
            [0.50000006 + 0.0j, 0.0 - 0.50000006j],
            [0.0 + 0.50000006j, 0.50000006 + 0.0j],
        ],
        dtype=numpy.complex64,
    )
    assert numpy.allclose(density_matrix, expected_density, rtol=1e-05)
    assert numpy.allclose(expected_density, tomo_res.data, rtol=0.09)
    assert numpy.allclose(density_matrix, tomo_res.data, rtol=0.09)
