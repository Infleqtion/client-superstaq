import copy
from typing import Dict, List, Tuple, Union

import cirq
import numpy as np
import numpy.typing as npt
import scipy.optimize as opt

import supermarq


class VQEProxy(supermarq.benchmark.Benchmark):
    """Proxy benchmark of a full VQE application that targets a single iteration
    of the whole variational optimization.

    The benchmark is parameterized by the number of qubits, n. For each value of
    n, we classically optimize the ansatz, sample 3 iterations near convergence,
    and use the sampled parameters to execute the corresponding circuits on the
    QPU. We take the measured energies from these experiments and average their
    values and compute a score based on how closely the experimental results are
    to the noiseless values.
    """

    def __init__(self, num_qubits: int, num_layers: int = 1) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.hamiltonian = self._gen_tfim_hamiltonian()
        self._params = self._gen_angles()

    def _gen_tfim_hamiltonian(self) -> List[Tuple[str, Union[int, Tuple[int, int]], int]]:
        r"""Generate an n-qubit Hamiltonian for a transverse-field Ising model (TFIM).

            $H = \sum_i^n(X_i) + \sum_i^n(Z_i Z_{i+1})$

        Example of a 6-qubit TFIM Hamiltonian:

            $H_6 = XIIIII + IXIIII + IIXIII + IIIXII + IIIIXI + IIIIIX + ZZIIII
                  + IZZIII + IIZZII + IIIZZI + IIIIZZ + ZIIIIZ$
        """
        hamiltonian: List[Tuple[str, Union[int, Tuple[int, int]], int]] = []
        for i in range(self.num_qubits):
            hamiltonian.append(("X", i, 1))  # [Pauli type, qubit idx, weight]
        for i in range(self.num_qubits - 1):
            hamiltonian.append(("ZZ", (i, i + 1), 1))
        hamiltonian.append(("ZZ", (self.num_qubits - 1, 0), 1))
        return hamiltonian

    def _gen_ansatz(self, params: npt.NDArray[np.float_]) -> List[cirq.Circuit]:
        qubits = cirq.LineQubit.range(self.num_qubits)
        z_circuit = cirq.Circuit()

        param_counter = 0
        for _ in range(self.num_layers):
            # Ry rotation block
            for i in range(self.num_qubits):
                z_circuit.append(cirq.Ry(rads=2 * params[param_counter])(qubits[i]))
                param_counter += 1
            # Rz rotation block
            for i in range(self.num_qubits):
                z_circuit.append(cirq.Rz(rads=2 * params[param_counter])(qubits[i]))
                param_counter += 1
            # Entanglement block
            for i in range(self.num_qubits - 1):
                z_circuit.append(cirq.CX(qubits[i], qubits[i + 1]))
            # Ry rotation block
            for i in range(self.num_qubits):
                z_circuit.append(cirq.Ry(rads=2 * params[param_counter])(qubits[i]))
                param_counter += 1
            # Rz rotation block
            for i in range(self.num_qubits):
                z_circuit.append(cirq.Rz(rads=2 * params[param_counter])(qubits[i]))
                param_counter += 1

        x_circuit = copy.deepcopy(z_circuit)
        x_circuit.append(cirq.H(q) for q in qubits)

        # Measure all qubits
        z_circuit.append(cirq.measure(*qubits))
        x_circuit.append(cirq.measure(*qubits))

        return [z_circuit, x_circuit]

    def _parity_ones(self, bitstr: str) -> int:
        one_count = 0
        for i in bitstr:
            if i == "1":
                one_count += 1
        return one_count % 2

    def _calc(self, bit_list: List[str], bitstr: str, probs: Dict[str, float]) -> float:
        energy = 0.0
        for item in bit_list:
            if self._parity_ones(item) == 0:
                energy += probs.get(bitstr, 0)
            else:
                energy -= probs.get(bitstr, 0)
        return energy

    def _get_expectation_value_from_probs(
        self, probs_z: Dict[str, float], probs_x: Dict[str, float]
    ) -> float:
        avg_energy = 0.0

        # Find the contribution to the energy from the X-terms: \sum_i{X_i}
        for bitstr in probs_x.keys():
            bit_list_x = [bitstr[i] for i in range(len(bitstr))]
            avg_energy += self._calc(bit_list_x, bitstr, probs_x)

        # Find the contribution to the energy from the Z-terms: \sum_i{Z_i Z_{i+1}}
        for bitstr in probs_z.keys():
            # fmt: off
            bit_list_z = [bitstr[(i - 1): (i + 1)] for i in range(1, len(bitstr))]
            # fmt: on
            bit_list_z.append(bitstr[0] + bitstr[-1])  # Add the wrap-around term manually
            avg_energy += self._calc(bit_list_z, bitstr, probs_z)

        return avg_energy

    def _get_opt_angles(self) -> Tuple[npt.NDArray[np.float_], float]:
        def f(params: npt.NDArray[np.float_]) -> float:
            """The objective function to minimize.

            Args:
                params: The parameters at which to evaluate the objective.

            Returns:
                Evaluation of objective given parameters.
            """
            z_circuit, x_circuit = self._gen_ansatz(params)
            z_probs = supermarq.simulation.get_ideal_counts(z_circuit)
            x_probs = supermarq.simulation.get_ideal_counts(x_circuit)
            energy = self._get_expectation_value_from_probs(z_probs, x_probs)

            return -energy  # because we are minimizing instead of maximizing

        init_params = [
            np.random.uniform() * 2 * np.pi for _ in range(self.num_layers * 4 * self.num_qubits)
        ]
        out = opt.minimize(f, init_params, method="COBYLA")

        return out["x"], out["fun"]

    def _gen_angles(self) -> npt.NDArray[np.float_]:
        """Classically simulate the variational optimization and return
        the final parameters.
        """
        params, _ = self._get_opt_angles()
        return params

    def circuit(self) -> List[cirq.Circuit]:
        """Construct a parameterized ansatz.

        The counts obtained from evaluating these two circuits should be passed to `score` in the
        same order they are returned here.

        Returns:
            A list of circuits for the VQE benchmark: the ansatz measured in the Z basis, and the
            ansatz measured in the X basis.
        """
        return self._gen_ansatz(self._params)

    def score(self, counts: List[Dict[str, float]]) -> float:
        """Compare the average energy measured by the experiments to the ideal value.

        The ideal value is obtained via noiseless simulation. In principle the ideal value can be
        obtained through efficient classical means since the 1D TFIM is analytically solvable.

        Args:
            counts: A dictionary containing the measurement counts from circuit execution.

        Returns:
            The VQE proxy benchmark score.
        """
        counts_z, counts_x = counts
        shots_z = sum(counts_z.values())
        probs_z = {bitstr: count / shots_z for bitstr, count in counts_z.items()}
        shots_x = sum(counts_x.values())
        probs_x = {bitstr: count / shots_x for bitstr, count in counts_x.items()}
        experimental_expectation = self._get_expectation_value_from_probs(probs_z, probs_x)

        circuit_z, circuit_x = self.circuit()
        ideal_expectation = self._get_expectation_value_from_probs(
            supermarq.simulation.get_ideal_counts(circuit_z),
            supermarq.simulation.get_ideal_counts(circuit_x),
        )

        return float(
            1.0 - abs(ideal_expectation - experimental_expectation) / abs(2 * ideal_expectation)
        )
