from typing import Dict, Iterator, List
import math

import cirq
from qiskit.quantum_info import hellinger_fidelity

from supermarq.benchmark import Benchmark


class QFT(Benchmark):
    """Creates a benchmark using both QFT and inverse-QFT.

    Implements Method 1 of the QFT bechmark from the QED-C
    benchmark suite: https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/quantum-fourier-transform/cirq/qft_benchmark.py
    """
    def __init__(self, num_qubits: int, secret_int: int):
        self.num_qubits = num_qubits
        self.secret_int = secret_int

    def _qft_gate(self, input_size):
        # allocate qubits
        qr = [cirq.GridQubit(i, 0) for i in range(input_size)]
        qc = cirq.Circuit()

        # Generate multiple groups of diminishing angle CRZs and H gate
        for i_qubit in range(0, input_size):

            # start laying out gates from highest order qubit (the hidx)
            hidx = input_size - i_qubit - 1

            # if not the highest order qubit, add multiple controlled RZs of decreasing angle
            if hidx < input_size - 1:
                num_crzs = i_qubit
                for j in range(0, num_crzs):
                    divisor = 2 ** (num_crzs - j)
                    qc.append(cirq.CZ(qr[hidx],qr[input_size - j - 1])**(1.0/divisor))

            # followed by an H gate (applied to all qubits)
            qc.append(cirq.H(qr[hidx]))

        return to_gate(num_qubits=input_size, circ=qc, name="qft")

    def _inv_qft_gate(self, input_size):
        """"Inverse QFT Circuit."""
        # allocate qubits
        qr = [cirq.GridQubit(i, 0) for i in range(input_size)]
        qc = cirq.Circuit()

        # Generate multiple groups of diminishing angle CRZs and H gate
        for i_qubit in reversed(range(0, input_size)):

            # start laying out gates from highest order qubit (the hidx)
            hidx = input_size - i_qubit - 1

            # precede with an H gate (applied to all qubits)
            qc.append(cirq.H(qr[hidx]))

            # if not the highest order qubit, add multiple controlled RZs of decreasing angle
            if hidx < input_size - 1:
                num_crzs = i_qubit
                for j in reversed(range(0, num_crzs)):
                    divisor = 2 ** (num_crzs - j)
                    qc.append(cirq.CZ(qr[hidx],qr[input_size - j - 1])**(-1.0/divisor))

        return to_gate(num_qubits=input_size, circ=qc, name="inv_qft")

    def circuit(self) -> cirq.Circuit:
        # Size of input is one less than available qubits
        input_size = self.num_qubits

        # allocate qubits
        qr = [cirq.GridQubit(i, 0) for i in range(self.num_qubits)]
        qc = cirq.Circuit()

        # Perform X on each qubit that matches a bit in secret string
        s = ('{0:0'+str(input_size)+'b}').format(self.secret_int)
        for i_qubit in range(input_size):
            if s[input_size-1-i_qubit]=='1':
                qc.append(cirq.X(qr[i_qubit]))

        # perform QFT on the input
        qc.append(self._qft_gate(input_size).on(*qr))

        # some compilers recognize the QFT and IQFT in series and collapse them to identity;
        # perform a set of rotations to add one to the secret_int to avoid this collapse
        for i_q in range(0, self.num_qubits):
            divisor = 2 ** (i_q)
            qc.append(cirq.rz( 1 * math.pi / divisor).on(qr[i_q]))

        # to revert back to initial state, apply inverse QFT
        qc.append(self._inv_qft_gate(input_size).on(*qr))

        # measure all qubits
        qc.append(cirq.measure(*[qr[i_qubit] for i_qubit in range(self.num_qubits)], key='result'))

        # return cirq circuit compiled to QASM
        return cirq.optimize_for_target_gateset(qc, gateset=cirq.CZTargetGateset())

    def score(self, counts: Dict[str, float]) -> float:
        ideal_state = ''.join(list(f"{self.secret_int + 1:0{self.num_qubits}b}")[::-1])
        return hellinger_fidelity(counts, {ideal_state: sum(counts.values())})


class to_gate(cirq.Gate):
    """Cirq utility class defined by QEDC"""
    def __init__(self, num_qubits, circ, name="G"):
        self.num_qubits=num_qubits
        self.circ = circ
        self.name = name

    def _num_qubits_(self):
        return self.num_qubits

    def _decompose_(self, qubits):
        # `sorted()` needed to correct error in `all_qubits()` not returning a reasonable order for all of the qubits
        qbs = sorted(list(self.circ.all_qubits()))
        mapping = {}
        for t in range(self.num_qubits):
            mapping[qbs[t]] = qubits[t]
        def f_map(q):
            return mapping[q]

        circ_new = self.circ.transform_qubits(f_map)
        return circ_new.all_operations()

    def _circuit_diagram_info_(self, args):
        return [self.name] * self._num_qubits_()

