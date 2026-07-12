import cirq
import numpy as np
import pytest

from cirq_superstaq import ft_compiler


def test_compile() -> None:
    qubit = cirq.LineQubit(0)
    logical_circuit = cirq.Circuit(cirq.X(qubit), cirq.X(qubit), cirq.measure(qubit, key="m"))

    compiled_circuit = ft_compiler.compile(logical_circuit)

    print(compiled_circuit)

    # assert len(compiled_circuit.all_qubits()) == 21
    # assert sum(op.gate == cirq.X for op in compiled_circuit.all_operations()) == 14
    # assert sum(isinstance(op.gate, cirq.ResetChannel) for op in compiled_circuit.all_operations()) == 14
    # assert cirq.measurement_key_names(compiled_circuit) == {
    #     "__ft_data_0_0",
    #     "__ft_ancilla_0_0",
    #     "__ft_data_0_1",
    #     "__ft_ancilla_0_1",
    #     "m",
    # }
    # measurement = next(
    #     op
    #     for op in compiled_circuit.all_operations()
    #     if isinstance(op.gate, cirq.MeasurementGate) and cirq.measurement_key_name(op) == "m"
    # )
    # assert measurement.qubits == tuple(cirq.LineQubit.range(7))


def test_compile_encodes_data_and_ancillas_in_parallel() -> None:
    qubit = cirq.LineQubit(0)
    logical_circuit = cirq.Circuit(cirq.I(qubit), cirq.measure(qubit, key="m"))

    compiled_circuit = ft_compiler.compile(logical_circuit)
    expected_encoding = cirq.Circuit.zip(
        ft_compiler.encode(cirq.LineQubit.range(0, 7)),
        ft_compiler.encode(cirq.LineQubit.range(7, 14)),
        ft_compiler.encode(cirq.LineQubit.range(14, 21)),
    )

    assert compiled_circuit[: len(expected_encoding)] == expected_encoding


def test_encode_and_decode() -> None:
    qubits = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.I(qubits[0]), cirq.measure(qubits[0], key="m"))

    simulator = ft_compiler.FTSimulator()
    result = simulator.run(circuit, repetitions=10)

    np.testing.assert_array_equal(result.measurements["m"], np.zeros((10, 1)))

def test_one_qubit_circuit():
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.X(qubits[0]), cirq.CX(qubits[0], qubits[1]), cirq.measure(qubits, key="m"))

    simulator = ft_compiler.FTSimulator()
    result = simulator.run(circuit, repetitions=10)
    print(result)


def test_hello_cirq():
    # Pick a qubit.
    qubits = cirq.LineQubit.range(1)

    # Create a circuit.
    circuit = cirq.Circuit(
    cirq.I(qubits[0]),
    cirq.measure(qubits[0], key='m')
)
    print("Circuit:")
    print(circuit)

    # Simulate the circuit several times.
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=20)
    print("Results:")
    print(result)

@pytest.mark.parametrize("error_index", range(7))
def test_syndrome_and_correct_error(error_index: int) -> None:
    measurement = np.zeros(7, dtype=np.int8)
    measurement[error_index] = 1

    corrected = ft_compiler.correct_error(measurement)

    np.testing.assert_array_equal(corrected, np.zeros(7))
    assert measurement[error_index] == 1  # The input is not mutated.


def test_generate_logical_circuit() -> None:
    data, ancilla0, ancilla1 = (
        cirq.LineQubit.range(start, start + 7) for start in (0, 7, 14)
    )
    circuit = ft_compiler.generate_logical_circuit(
        cirq.X.on_each(*data),
        data[0],
        data,
        ancilla0,
        ancilla1,
        error_probability=0.0,
    )

    result = ft_compiler.qec_simulator(circuit, repetitions=5)

    np.testing.assert_array_equal(result.measurements["teleported_measurements"], np.ones((5, 1)))


def test_validation() -> None:
    with pytest.raises(ValueError, match="must contain 7 qubits"):
        ft_compiler.encode(cirq.LineQubit.range(6))

    blocks = [cirq.LineQubit.range(start, start + 7) for start in (0, 7, 14)]
    with pytest.raises(ValueError, match="must belong to the data block"):
        ft_compiler.generate_logical_circuit([], cirq.LineQubit(100), *blocks)
