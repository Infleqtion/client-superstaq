import cirq
import numpy as np

import cirq_superstaq.custom_gates


def test_fermionic_swap_gate() -> None:
    theta = 0.123
    gate = cirq_superstaq.custom_gates.FermionicSWAPGate(theta)

    assert str(gate) == "FermionicSWAPGate(0.123)"
    assert repr(gate) == "cirq_superstaq.custom_gates.FermionicSWAPGate(0.123)"
    cirq.testing.assert_equivalent_repr(gate, setup_code="import cirq_superstaq.custom_gates")

    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(gate(qubits[0], qubits[2]))

    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───FermionicSWAP(0.0392π)───
      │
2: ───FermionicSWAP(0.0392π)───
""",
    )

    expected = np.array(
        [[1, 0, 0, 0], [0, 0, np.exp(1j * theta), 0], [0, np.exp(1j * theta), 0, 0], [0, 0, 0, 1]]
    )
    assert np.allclose(cirq.unitary(gate), expected)


def test_custom_resolver() -> None:
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(2)
    circuit += cirq_superstaq.custom_gates.FermionicSWAPGate(1.23)(*qubits)
    circuit += cirq.CX(*qubits)

    json_text = cirq.to_json(circuit)
    resolvers = [cirq_superstaq.custom_gates.custom_resolver, *cirq.DEFAULT_RESOLVERS]
    assert cirq.read_json(json_text=json_text, resolvers=resolvers) == circuit
