import io
import warnings
from unittest import mock

import general_superstaq as gss
import pytest
import qiskit

import qiskit_superstaq as qss


def test_assign_unique_inst_names() -> None:
    inst_0 = qss.ZZSwapGate(0.1)
    inst_1 = qss.ZZSwapGate(0.2)
    inst_2 = qss.ZZSwapGate(0.1)

    circuit = qiskit.QuantumCircuit(4)
    circuit.append(inst_0, [0, 1])
    circuit.append(inst_1, [1, 2])
    circuit.append(inst_2, [2, 0])
    circuit.append(inst_1, [0, 1])
    circuit.rxx(1.1, 0, 1)
    circuit.rxx(1.2, 1, 2)

    expected_inst_names = [
        "zzswap",
        "zzswap_1",
        "zzswap",
        "zzswap_1",
        "rxx",
        "rxx",
    ]

    new_circuit = qss.serialization._assign_unique_inst_names(circuit)
    assert [inst.name for inst, _, _ in new_circuit] == expected_inst_names


def test_circuit_serialization() -> None:
    circuit_0 = qiskit.QuantumCircuit(3)
    circuit_0.cx(2, 1)
    circuit_0.rz(0.86, 0)

    circuit_1 = qiskit.QuantumCircuit(4)
    circuit_1.append(qss.AceCR("+-"), [0, 1])
    circuit_1.append(qss.AceCR("-+"), [1, 2])
    circuit_1.append(qss.AceCR("+-", 1.23), [0, 1])
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


def test_warning_suppression() -> None:
    circuit = qiskit.QuantumCircuit(3)
    circuit.cx(2, 1)
    circuit.h(0)

    major, minor, patch = qiskit.__version__.split(".")
    newer_version = f"{major}.{minor}.{int(patch) + 1}"

    # QPY encodes qiskit.__version__ into the serialized circuit, so mocking a newer version string
    # during serialization will cause a QPY version UserWarning during deserialization
    with mock.patch("qiskit.qpy.interface.__version__", newer_version):
        serialized_circuit = qss.serialization.serialize_circuits(circuit)

    # Check that a warning would normally be thrown
    with pytest.warns(UserWarning):
        buf = io.BytesIO(gss.converters._str_to_bytes(serialized_circuit))
        _ = qiskit.qpy.load(buf)

    # Check that it is suppressed by deserialize_circuits
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        _ = qss.serialization.deserialize_circuits(serialized_circuit)
