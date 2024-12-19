# pylint: disable=missing-function-docstring,missing-class-docstring
from __future__ import annotations

import importlib
import io
import json
import warnings
from unittest import mock

import general_superstaq as gss
import numpy as np
import pytest
import qiskit
import qiskit.qasm2

import qiskit_superstaq as qss


def test_qpy_serialization_version() -> None:
    assert (
        qiskit.qpy.common.QPY_COMPATIBILITY_VERSION
        <= qss.serialization.QPY_SERIALIZATION_VERSION
        <= qiskit.qpy.common.QPY_VERSION
    )


def test_to_json() -> None:
    real_part = np.random.uniform(-1, 1, size=(4, 4))
    imag_part = np.random.uniform(-1, 1, size=(4, 4))

    val = [
        {"abc": 123},
        real_part,
        real_part + 1j * imag_part,
    ]

    json_str = qss.serialization.to_json(val)
    assert json.loads(json_str) == [
        {
            "abc": 123,
        },
        {
            "type": "qss_array",
            "real": real_part.tolist(),
            "imag": 4 * [[0, 0, 0, 0]],
        },
        {
            "type": "qss_array",
            "real": real_part.tolist(),
            "imag": imag_part.tolist(),
        },
    ]
    resolved_val = json.loads(json_str, object_hook=qss.serialization.json_resolver)
    assert resolved_val[0] == val[0]
    assert isinstance(resolved_val[1], np.ndarray)
    assert isinstance(resolved_val[2], np.ndarray)
    assert np.all(resolved_val[1] == val[1])
    assert np.all(resolved_val[2] == val[2])

    with pytest.raises(TypeError, match="not JSON serializable"):
        qss.serialization.to_json(qiskit.QuantumCircuit())


def test_circuit_serialization() -> None:
    circuit_0 = qiskit.QuantumCircuit(3)
    circuit_0.cx(2, 1)
    circuit_0.rz(0.86, 0)

    circuit_1 = qiskit.QuantumCircuit(4)
    circuit_1.append(qss.AceCR("+-"), [0, 1])
    circuit_1.append(qss.AceCR("-+"), [1, 2])
    circuit_1.append(qss.AceCR("+-", sandwich_rx_rads=1.23), [0, 1])
    circuit_1.append(qss.AceCR(np.pi / 3), [0, 1])
    circuit_1.append(qss.ZZSwapGate(0.75), [2, 0])
    circuit_1.append(qss.AQTiCCXGate(), [0, 1, 2])
    circuit_1.append(qss.custom_gates.iCCXGate(), [0, 1, 2])
    circuit_1.append(qss.custom_gates.iCCXGate(ctrl_state="10"), [0, 1, 2])
    circuit_1.append(qss.custom_gates.iCCXGate(ctrl_state="01"), [0, 1, 2])
    circuit_1.append(qss.custom_gates.iCCXdgGate(), [0, 1, 2])
    circuit_1.append(qss.custom_gates.iCCXdgGate(ctrl_state="10"), [0, 1, 2])
    circuit_1.append(qss.custom_gates.iCCXdgGate(ctrl_state="01"), [0, 1, 2])
    circuit_1.append(qss.custom_gates.iCCXdgGate(ctrl_state="00"), [0, 1, 2])
    circuit_1.append(qss.custom_gates.iXGate(), [2])
    circuit_1.append(qss.custom_gates.iXdgGate(), [2])

    circuit_1.append(
        qss.ParallelGates(
            qss.ZZSwapGate(3.09),
            qiskit.circuit.library.RXXGate(0.1),
        ),
        [1, 2, 0, 3],
    )
    circuit_1.append(
        qss.ParallelGates(
            qss.ZZSwapGate(3.09),
            qiskit.circuit.library.RXXGate(0.2),
        ),
        [0, 3, 2, 1],
    )
    circuit_1.append(
        qss.ParallelGates(
            qss.ZZSwapGate(3.09),
            qiskit.circuit.library.RXXGate(0.1),
        ),
        [0, 1, 3, 2],
    )

    serialized_circuit = qss.serialization.serialize_circuits(circuit_0)
    assert isinstance(serialized_circuit, str)
    assert qss.serialization.deserialize_circuits(serialized_circuit) == [circuit_0]

    circuits = [circuit_0, circuit_1]
    serialized_circuits = qss.serialization.serialize_circuits(circuits)
    assert isinstance(serialized_circuits, str)
    assert qss.serialization.deserialize_circuits(serialized_circuits) == circuits


def test_insert_times_and_durations() -> None:
    circuit = qiskit.QuantumCircuit(2)

    # Test an empty circuit
    new_circuit = qss.serialization.insert_times_and_durations(circuit, [], [])
    assert new_circuit.duration == 0
    assert new_circuit.op_start_times == []

    circuit.x(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    durations = [10, 100, 0, 50, 50]
    start_times = [0, 10, 110, 110, 110]

    new_circuit = qss.serialization.insert_times_and_durations(circuit, durations, start_times)
    assert new_circuit == circuit
    assert new_circuit.op_start_times == start_times
    assert [inst.operation.duration for inst in new_circuit] == durations
    assert new_circuit.duration == 160

    # Test fallback when timing information is not provided
    new_circuit = qss.serialization.insert_times_and_durations(circuit, [], [])
    assert new_circuit is circuit
    assert new_circuit.duration is None


def test_warning_suppression() -> None:
    circuit = qiskit.QuantumCircuit(3)
    circuit.cx(2, 1)
    circuit.h(0)

    major, minor, patch, *_ = qiskit.__version__.split(".")
    newer_version = f"{major}.{minor}.{int(patch) + 1}"

    # QPY encodes qiskit.__version__ into the serialized circuit, so mocking a newer version string
    # during serialization will cause a QPY version UserWarning during deserialization
    with mock.patch("qiskit.qpy.interface.__version__", newer_version):
        serialized_circuit = qss.serialization.serialize_circuits(circuit)

    # Check that a warning would normally be thrown
    with pytest.warns(UserWarning):
        buf = io.BytesIO(gss.serialization.str_to_bytes(serialized_circuit))
        _ = qiskit.qpy.load(buf)

    # Check that it is suppressed by deserialize_circuits
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        _ = qss.serialization.deserialize_circuits(serialized_circuit)


def test_deserialization_errors() -> None:
    circuit = qiskit.QuantumCircuit(3)
    circuit.x(0)

    # Mock a circuit serialized with a newer version of QPY:
    with mock.patch("qiskit_superstaq.serialization.QPY_SERIALIZATION_VERSION", None):
        with mock.patch("qiskit.qpy.common.QPY_VERSION", qiskit.qpy.common.QPY_VERSION + 1):
            serialized_circuit = qss.serialize_circuits(circuit)

    # Remove a few bytes to force a deserialization error
    serialized_circuit = gss.serialization.bytes_to_str(
        gss.serialization.str_to_bytes(serialized_circuit)[:-4]
    )

    with pytest.raises(ValueError, match="your version of Qiskit"):
        with mock.patch("qiskit.qpy.common.QPY_VERSION", qiskit.qpy.common.QPY_VERSION - 3):
            _ = qss.deserialize_circuits(serialized_circuit)

    # Mock a circuit serialized with an older of QPY:
    with mock.patch("qiskit_superstaq.serialization.QPY_SERIALIZATION_VERSION", None):
        with mock.patch("qiskit.qpy.common.QPY_VERSION", 3):
            serialized_circuit = qss.serialize_circuits(circuit)

    with pytest.raises(ValueError, match="Please contact"):
        _ = qss.deserialize_circuits(serialized_circuit)


def test_circuit_from_qasm_with_gate_defs() -> None:
    # Regression test: Calling `operation._define()` breaks gates generated from QASM definitions.
    circuit = qiskit.circuit.library.QuantumVolume(4)
    circuit.measure_all()

    new_circuit = qss.deserialize_circuits(qss.serialize_circuits(circuit))[0]
    assert circuit == new_circuit

    circuit_from_qasm = qiskit.QuantumCircuit.from_qasm_str(qiskit.qasm2.dumps(circuit))
    new_circuit = qss.deserialize_circuits(qss.serialize_circuits(circuit_from_qasm))[0]

    # QASM conversion can change instruction names, so compare by unitary
    new_circuit.remove_final_measurements()
    circuit_from_qasm.remove_final_measurements()
    assert np.allclose(
        qiskit.quantum_info.Operator(new_circuit).to_matrix(),
        qiskit.quantum_info.Operator(circuit_from_qasm).to_matrix(),
        atol=1e-8,
    )


# Gate classes and corresponding numbers of parameters
test_gates = {
    qiskit.circuit.library.Measure: 0,
    qiskit.circuit.library.Reset: 0,
    qiskit.circuit.library.DCXGate: 0,
    qiskit.circuit.library.ECRGate: 0,
    qiskit.circuit.library.HGate: 0,
    qiskit.circuit.library.GlobalPhaseGate: 1,
    qiskit.circuit.library.PhaseGate: 1,
    qiskit.circuit.library.RC3XGate: 0,
    qiskit.circuit.library.RCCXGate: 0,
    qiskit.circuit.library.RGate: 2,
    qiskit.circuit.library.RVGate: 3,
    qiskit.circuit.library.RXGate: 1,
    qiskit.circuit.library.RXXGate: 1,
    qiskit.circuit.library.RYGate: 1,
    qiskit.circuit.library.RYYGate: 1,
    qiskit.circuit.library.RZGate: 1,
    qiskit.circuit.library.RZXGate: 1,
    qiskit.circuit.library.RZZGate: 1,
    qiskit.circuit.library.SGate: 0,
    qiskit.circuit.library.SXGate: 0,
    qiskit.circuit.library.SXdgGate: 0,
    qiskit.circuit.library.SdgGate: 0,
    qiskit.circuit.library.SwapGate: 0,
    qiskit.circuit.library.TGate: 0,
    qiskit.circuit.library.TdgGate: 0,
    qiskit.circuit.library.U1Gate: 1,
    qiskit.circuit.library.U2Gate: 2,
    qiskit.circuit.library.U3Gate: 3,
    qiskit.circuit.library.UGate: 3,
    qiskit.circuit.library.XGate: 0,
    qiskit.circuit.library.XXMinusYYGate: 2,
    qiskit.circuit.library.XXPlusYYGate: 2,
    qiskit.circuit.library.YGate: 0,
    qiskit.circuit.library.ZGate: 0,
    qiskit.circuit.library.iSwapGate: 0,
    qiskit.circuit.library.C3SXGate: 0,
    qiskit.circuit.library.C3XGate: 0,
    qiskit.circuit.library.C4XGate: 0,
    qiskit.circuit.library.CCXGate: 0,
    qiskit.circuit.library.CCZGate: 0,
    qiskit.circuit.library.CHGate: 0,
    qiskit.circuit.library.CXGate: 0,
    qiskit.circuit.library.CYGate: 0,
    qiskit.circuit.library.CZGate: 0,
    qiskit.circuit.library.CSGate: 0,
    qiskit.circuit.library.CSXGate: 0,
    qiskit.circuit.library.CSdgGate: 0,
    qiskit.circuit.library.CSwapGate: 0,
    qiskit.circuit.library.CPhaseGate: 1,
    qiskit.circuit.library.CRXGate: 1,
    qiskit.circuit.library.CRYGate: 1,
    qiskit.circuit.library.CRZGate: 1,
    qiskit.circuit.library.CU1Gate: 1,
    qiskit.circuit.library.CU3Gate: 3,
    qiskit.circuit.library.CUGate: 4,
    qiskit.circuit.library.IGate: 0,
    qss.AceCR: 2,
    qss.AQTiCCXGate: 0,
    qss.DDGate: 1,
    qss.StrippedCZGate: 1,
    qss.ZZSwapGate: 1,
    qss.custom_gates.iXGate: 0,
    qss.custom_gates.iXdgGate: 0,
    qss.custom_gates.iCCXGate: 0,
    qss.custom_gates.iCCXdgGate: 0,
}


def test_completeness() -> None:
    """Makes sure `qss.serialization._custom_gates_by_name` and tests cover all custom gates."""
    for attr_name in dir(qss.custom_gates):
        attr = getattr(qss.custom_gates, attr_name)
        if isinstance(attr, type) and issubclass(attr, qiskit.circuit.Instruction):
            assert issubclass(
                attr, tuple(qss.serialization._custom_gates_by_name.values())
            ), f"'{attr_name}' not covered in `qss.serialization._custom_gates_by_name`."

            if attr is not qss.ParallelGates:
                assert attr in test_gates, f"'{attr_name}' missing from `test_gates`."


def test_mcphase() -> None:
    gate1 = qss.serialization._mcphase(1.1, 3, ctrl_state=2)
    with mock.patch("qiskit.__version__", "1.0.2"):
        gate3 = qss.serialization._mcphase(1.1, 3, ctrl_state=2)
    with mock.patch("qiskit.__version__", "1.1.1"), mock.patch(
        "qiskit.circuit.library.MCPhaseGate", return_value=gate1
    ):
        gate2 = qss.serialization._mcphase(1.1, 3, ctrl_state=2)
    assert gate1 == gate2 == gate3


def test_qft_gate() -> None:
    # This test is necessary to get full coverage with Qiskit < 1.1.0
    try:
        with mock.patch(
            "qiskit.circuit.library.QFTGate", qiskit.circuit.library.Barrier, create=True
        ) as mock_qft:
            importlib.reload(qss.serialization)
            assert mock_qft in qss.serialization._controlled_gate_resolvers
    finally:
        importlib.reload(qss.serialization)  # Reset to avoid side effects


@pytest.mark.parametrize("base_class", test_gates, ids=lambda g: g.name)
def test_gate_preparation_and_resolution(base_class: type[qiskit.circuit.Instruction]) -> None:
    num_params = test_gates[base_class]

    gate = base_class(*np.random.uniform(-2 * np.pi, 2 * np.pi, num_params))
    assert qss.serialization._resolve_gate(qss.serialization._prepare_gate(gate)) == gate
    assert qss.serialization._resolve_gate(qss.serialization._wrap_gate(gate)) == gate


def _check_serialization(*gates: qiskit.circuit.Instruction) -> None:
    num_qubits = max(
        [
            sum(g.num_qubits for g in gates),
            *(
                g.num_qubits + 1
                for g in gates
                if not issubclass(g.base_class, qiskit.circuit.ControlledGate)
            ),
        ]
    )

    circuit = qiskit.QuantumCircuit(num_qubits, max((g.num_clbits for g in gates), default=0))

    for gate in gates:
        circuit.append(gate, range(gate.num_qubits), range(gate.num_clbits))

        if isinstance(gate, qiskit.circuit.Gate) and not isinstance(
            gate, qiskit.circuit.ControlledGate
        ):
            circuit.append(gate.control(1, ctrl_state=0), range(gate.num_qubits + 1))
            circuit.append(gate.control(1), range(gate.num_qubits + 1))

    # Make sure resolution recurses into component gates
    if all(isinstance(gate, qiskit.circuit.Gate) for gate in gates):
        parallel_gates = qss.ParallelGates(*gates)
        circuit.append(qss.ParallelGates(*gates), range(parallel_gates.num_qubits))

    # Make sure resolution recurses into control-flow operations
    circuit.for_loop([0, 1], body=circuit.copy(), qubits=circuit.qubits, clbits=circuit.clbits)

    # Make sure resolution recurses into sub-operations
    subcircuit = circuit.copy()
    subcircuit.append(subcircuit.to_instruction(), subcircuit.qubits, subcircuit.clbits)
    circuit.append(subcircuit, circuit.qubits, circuit.clbits)

    new_circuit = qss.deserialize_circuits(qss.serialize_circuits(circuit))[0]
    assert circuit == new_circuit

    # Try one more round
    assert circuit == qss.deserialize_circuits(qss.serialize_circuits(new_circuit))[0]

    assert all(
        inst1.operation.base_class == inst2.operation.base_class
        for inst1, inst2 in zip(circuit, new_circuit)
        if inst1.operation.base_class is not qss.AQTiCCXGate
    )


@pytest.mark.parametrize("base_class", test_gates, ids=lambda g: g.name)
def test_gate_serialization(base_class: type[qiskit.circuit.Instruction]) -> None:
    num_params = test_gates[base_class]
    params = np.random.uniform(-2 * np.pi, 2 * np.pi, (2, num_params))

    # Construct two different gates to test https://github.com/Qiskit/qiskit/issues/8941 workaround
    gate1 = base_class(*params[0])
    if issubclass(base_class, qiskit.circuit.ControlledGate) and base_class != qss.AQTiCCXGate:
        gate2 = base_class(*params[1], ctrl_state=gate1.ctrl_state // 2)
    else:
        gate2 = base_class(*params[1])

    _check_serialization(gate1, gate2)


@pytest.mark.parametrize(
    "gate",
    [
        qiskit.circuit.library.MSGate(2, np.random.uniform(-2 * np.pi, 2 * np.pi)),
        qiskit.circuit.library.MSGate(3, np.random.uniform(-2 * np.pi, 2 * np.pi)),
        qiskit.circuit.library.MCXGrayCode(4),
        qiskit.circuit.library.MCXGate(3),
        qiskit.circuit.library.MCXGate(5),
        qiskit.circuit.library.MCU1Gate(np.random.uniform(-2 * np.pi, 2 * np.pi), 3),
        qiskit.circuit.library.MCPhaseGate(np.random.uniform(-2 * np.pi, 2 * np.pi), 3),
        *(
            [qiskit.circuit.library.QFTGate(4)]
            if hasattr(qiskit.circuit.library, "QFTGate")
            else []
        ),
    ],
    ids=lambda g: g.name,
)
def test_nonstandard_gate_serialization(gate: qiskit.circuit.Instruction) -> None:
    gates = [gate]
    if hasattr(gate, "ctrl_state"):
        gate2 = gate.copy().to_mutable()
        gate2.ctrl_state = 1
        gates.append(gate2)

    _check_serialization(*gates)


def test_qiskit_gate_workarounds() -> None:
    """Tests workarounds for qiskit gates which are not handled correctly by QPY."""
    _check_serialization(
        qiskit.circuit.library.MSGate(2, 1.1),
        qiskit.circuit.library.MSGate(2, 2.2),
        qiskit.circuit.library.MSGate(3, 2.2),
    )

    circuit = qiskit.QuantumCircuit(6)
    circuit.append(qiskit.circuit.library.MCXGate(3, ctrl_state=1), range(4))
    circuit.append(qiskit.circuit.library.MCXGate(5, ctrl_state=6), range(6))
    circuit.append(qiskit.circuit.library.MCXGrayCode(4, ctrl_state=2), range(5))
    circuit.append(qiskit.circuit.library.MCU1Gate(1.1, 3, ctrl_state=5), range(4))
    circuit.append(qiskit.circuit.library.MCU1Gate(2.2, 3, ctrl_state=6), range(4))
    circuit.append(qiskit.circuit.library.MCPhaseGate(1.1, 3), range(4))
    circuit.append(qiskit.circuit.library.MCPhaseGate(2.2, 3), range(4))

    subcircuit = circuit.copy()
    subcircuit.append(circuit, circuit.qubits)
    circuit.append(subcircuit, circuit.qubits)

    assert qss.deserialize_circuits(qss.serialize_circuits(circuit))[0] == circuit


def test_different_gates_with_same_name() -> None:
    gate1 = qss.ZZSwapGate(1.2)
    gate2 = qss.AceCR(1.2, sandwich_rx_rads=np.pi / 2).copy(gate1.name)
    gate3 = gate1.copy()
    gate3.definition = gate2.definition

    circuit = qiskit.QuantumCircuit(2)
    circuit.append(gate1, [0, 1])
    circuit.append(gate2, [0, 1])
    circuit.append(gate3, [0, 1])

    expected_gate2 = qiskit.circuit.Gate(gate2.name, gate2.num_qubits, gate2.params)
    expected_gate2.definition = gate2.definition
    expected_gate3 = qiskit.circuit.Gate(gate3.name, gate3.num_qubits, gate3.params)
    expected_gate3.definition = gate3.definition

    expected_circuit = qiskit.QuantumCircuit(2)
    expected_circuit.append(gate1, [0, 1])
    expected_circuit.append(expected_gate2, [0, 1])
    expected_circuit.append(expected_gate3, [0, 1])

    assert qss.deserialize_circuits(qss.serialize_circuits(circuit))[0] == expected_circuit
