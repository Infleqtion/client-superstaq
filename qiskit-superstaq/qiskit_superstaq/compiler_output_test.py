# pylint: disable=missing-function-docstring,missing-class-docstring
from __future__ import annotations

import importlib
import json
import pickle
import textwrap
from unittest import mock

import general_superstaq as gss
import pytest
import qiskit

import qiskit_superstaq as qss


def test_active_qubit_indices() -> None:
    circuit = qiskit.QuantumCircuit(4)
    circuit.add_register(qiskit.QuantumRegister(2, "foo"))

    circuit.x(3)
    circuit.cz(3, 5)
    circuit.barrier(0, 1, 2, 3, 4, 5)
    circuit.h(circuit.qubits[1])

    assert qss.active_qubit_indices(circuit) == [1, 3, 5]


def test_measured_qubit_indices() -> None:
    circuit = qiskit.QuantumCircuit(8, 2)
    circuit.x(0)
    circuit.measure(1, 0)
    circuit.cx(1, 2)
    circuit.measure([6, 5], [0, 1])
    circuit.measure([1, 3], [0, 1])  # (qubit 1 was already measured)
    circuit.measure([5, 1], [0, 1])  # (both were already measured)
    assert qss.measured_qubit_indices(circuit) == [1, 3, 5, 6]

    # Check that measurements in custom gates/subcircuits are handled correctly
    circuit = qiskit.QuantumCircuit(6, 2)
    circuit.measure(0, 0)
    assert qss.measured_qubit_indices(circuit) == [0]

    subcircuit = qiskit.QuantumCircuit(6, 2)
    subcircuit.x(0)
    subcircuit.measure(1, 0)
    subcircuit.measure(2, 1)
    assert qss.measured_qubit_indices(subcircuit) == [1, 2]

    # Append subcircuit to itself (measurements should land on qubits 2 and 4)
    subcircuit.append(subcircuit, [3, 2, 4, 0, 1, 5], [1, 0])
    assert qss.measured_qubit_indices(subcircuit) == [1, 2, 4]

    # Append subcircuit to circuit (measurements should land on qubits 4, 3, and 1 of circuit)
    circuit.append(subcircuit, [5, 4, 3, 2, 1, 0], [0, 1])
    assert qss.measured_qubit_indices(circuit) == [0, 1, 3, 4]


def test_measured_clbit_indices() -> None:
    # Test len(qiskit.ClassicalRegister()) < len(qiskit.QuantumRegister())
    circuit = qiskit.QuantumCircuit(8, 2)
    circuit.x(0)
    circuit.measure(1, 0)
    circuit.cx(1, 2)
    circuit.measure([6, 5], [0, 1])
    circuit.measure([1, 3], [0, 1])
    circuit.measure([5, 1], [0, 1])
    assert qss.classical_bit_mapping(circuit) == {0: 5, 1: 1}

    # Test len(qiskit.ClassicalRegister()) > len(qiskit.QuantumRegister())
    circuit = qiskit.QuantumCircuit(3, 5)
    circuit.h(1)
    circuit.x(2)
    circuit.measure([0, 1, 2], [2, 4, 1])
    assert qss.classical_bit_mapping(circuit) == {2: 0, 4: 1, 1: 2}

    # Test len(qiskit.ClassicalRegister()) = len(qiskit.QuantumRegister())
    circuit = qiskit.QuantumCircuit(9, 9)
    circuit.h(1)
    circuit.x(4)
    circuit.s(1)
    circuit.cx(1, 0)
    circuit.measure([0, 1, 4], [0, 1, 2])
    assert qss.classical_bit_mapping(circuit) == {0: 0, 1: 1, 2: 4}

    circuit.measure([0, 1, 4, 2, 3, 5, 6, 7], [0, 1, 2, 8, 7, 6, 5, 3])
    assert qss.classical_bit_mapping(circuit) == {0: 0, 1: 1, 2: 4, 8: 2, 7: 3, 6: 5, 5: 6, 3: 7}

    # Custom instruction with measurements test
    circuit_instr = qiskit.QuantumCircuit(2, 2)
    circuit_instr.h(0)
    circuit_instr.x(0)
    circuit_instr.measure([0, 1], [0, 1])
    custom_instruction = circuit_instr.to_instruction()

    circuit = qiskit.QuantumCircuit(4, 4)
    circuit.append(custom_instruction, [1, 2], [2, 3])

    assert qss.classical_bit_mapping(circuit) == {2: 1, 3: 2}

    circuit.append(custom_instruction, [2, 1], [2, 3])
    assert qss.classical_bit_mapping(circuit) == {2: 2, 3: 1}


def test_compiler_output_repr() -> None:
    circuit = qiskit.QuantumCircuit(4)
    assert (
        repr(qss.compiler_output.CompilerOutput(circuit, {0: 0}, {0: 1}))
        == f"""CompilerOutput({circuit!r}, {{0: 0}}, {{0: 1}}, None, None, None)"""
    )

    circuits = [circuit, circuit]
    assert (
        repr(qss.compiler_output.CompilerOutput(circuits, [{0: 0}, {1: 1}], [{0: 1}, {1: 0}]))
        == f"CompilerOutput({circuits!r}, [{{0: 0}}, {{1: 1}}], [{{0: 1}}, {{1: 0}}], None, "
        f"None, None)"
    )


def test_read_json() -> None:
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    qc_pulse = qc.copy()
    qc_pulse.add_calibration("cx", [0, 1], qiskit.pulse.ScheduleBlock("foo"))

    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "initial_logical_to_physicals": "[[]]",
        "final_logical_to_physicals": "[[]]",
        "pulse_gate_circuits": qss.serialization.serialize_circuits(qc_pulse),
        "pulse_durations": [[10, 20]],
        "pulse_start_times": [[0, 10]],
    }

    out = qss.compiler_output.read_json(json_dict, circuits_is_list=False)
    assert out.circuit == qc
    assert isinstance(out.pulse_gate_circuit, qiskit.QuantumCircuit)
    assert out.pulse_gate_circuit == qc_pulse
    assert out.pulse_gate_circuit.duration == 30
    assert out.pulse_gate_circuit[0].operation.duration == 10
    assert out.pulse_gate_circuit.op_start_times == [0, 10]

    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc, qc]),
        "initial_logical_to_physicals": "[[]]",
        "final_logical_to_physicals": "[[], []]",
        "pulse_gate_circuits": qss.serialization.serialize_circuits([qc_pulse, qc_pulse]),
        "pulse_durations": [[10, 20], [100, 200]],
        "pulse_start_times": [[0, 10], [0, 100]],
    }
    out = qss.compiler_output.read_json(json_dict, circuits_is_list=True)
    assert out.circuits == [qc, qc]
    assert out.pulse_gate_circuits == [qc_pulse, qc_pulse]
    assert out.pulse_gate_circuits[0].duration == 30
    assert out.pulse_gate_circuits[1].duration == 300
    assert out.pulse_gate_circuits[1][0].operation.duration == 100
    assert out.pulse_gate_circuits[1].op_start_times == [0, 100]

    json_dict["pulses"] = "oops"
    out = qss.compiler_output.read_json(json_dict, circuits_is_list=True)
    assert out.circuits == [qc, qc]


def test_read_json_empty_circuit() -> None:
    """Checks for common bugs when deserializing empty circuits."""
    qc = qiskit.QuantumCircuit(2)

    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "initial_logical_to_physicals": "[[]]",
        "final_logical_to_physicals": "[[]]",
        "pulse_gate_circuits": qss.serialization.serialize_circuits(qc),
        "pulse_durations": [[]],
        "pulse_start_times": [[]],
    }

    out = qss.compiler_output.read_json(json_dict, circuits_is_list=False)
    assert out.circuit == qc
    assert isinstance(out.pulse_gate_circuit, qiskit.QuantumCircuit)
    assert out.pulse_gate_circuit == qc
    assert out.pulse_gate_circuit.duration == 0
    assert out.pulse_gate_circuit.op_start_times == []

    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc, qc]),
        "initial_logical_to_physicals": "[[]]",
        "final_logical_to_physicals": "[[], []]",
        "pulse_gate_circuits": qss.serialization.serialize_circuits([qc, qc]),
        "pulse_durations": [[], []],
        "pulse_start_times": [[], []],
    }
    out = qss.compiler_output.read_json(json_dict, circuits_is_list=True)
    assert out.circuits == [qc, qc]
    assert out.pulse_gate_circuits == [qc, qc]
    assert out.pulse_gate_circuits[0].duration == 0
    assert out.pulse_gate_circuits[1].duration == 0
    assert out.pulse_gate_circuits[1].op_start_times == []


@mock.patch.dict("sys.modules", {"qtrl": None})
def test_read_json_aqt() -> None:
    importlib.reload(qss.compiler_output)

    circuit = qiskit.QuantumCircuit(4)
    for i in range(4):
        circuit.h(i)

    state_str = gss.serialization.serialize({})

    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits(circuit),
        "initial_logical_to_physicals": "[[]]",
        "final_logical_to_physicals": "[[]]",
        "state_jp": state_str,
    }

    with pytest.warns(UserWarning, match="deserialize compiled pulse sequences"):
        out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=False)

    assert out.circuit == circuit
    assert not hasattr(out, "circuits")

    with pytest.warns(UserWarning, match="deserialize compiled pulse sequences"):
        out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)

    assert out.circuits == [circuit]
    assert not hasattr(out, "circuit")

    # multiple circuits
    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits([circuit, circuit]),
        "initial_logical_to_physicals": "[[]]",
        "final_logical_to_physicals": "[[], []]",
        "state_jp": state_str,
    }

    with pytest.warns(UserWarning, match="deserialize compiled pulse sequences"):
        out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)

    assert out.circuits == [circuit, circuit]
    assert not hasattr(out, "circuit")

    # no sequence returned
    json_dict.pop("state_jp")
    out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)
    assert out.seq is None


def test_read_json_with_qtrl() -> None:  # pragma: no cover, b/c test requires qtrl installation
    qtrl = pytest.importorskip("qtrl", reason="qtrl not installed")
    seq = qtrl.sequencer.Sequence(n_elements=1)

    circuit = qiskit.QuantumCircuit(4)
    for i in range(4):
        circuit.h(i)
    circuit.measure_all()

    state_str = gss.serialization.serialize(seq.__getstate__())
    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits(circuit),
        "initial_logical_to_physicals": "[[]]",
        "final_logical_to_physicals": "[[]]",
        "state_jp": state_str,
    }

    out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert isinstance(out.seq, qtrl.sequencer.Sequence)
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert not hasattr(out.seq, "_readout")
    assert not hasattr(out, "circuits")

    # Serialized readout attribute for aqt_zurich_qpu:
    json_dict["readout_jp"] = state_str
    json_dict["readout_qubits"] = "[4, 5, 6, 7]"
    out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert isinstance(out.seq, qtrl.sequencer.Sequence)
    assert isinstance(out.seq._readout, qtrl.sequencer.Sequence)
    assert isinstance(out.seq._readout._readout, qtrl.sequence_utils.readout._ReadoutInfo)
    assert out.seq._readout._readout.sequence is out.seq._readout
    assert out.seq._readout._readout.qubits == [4, 5, 6, 7]
    assert out.seq._readout._readout.n_readouts == 1
    assert pickle.dumps(out.seq._readout) == pickle.dumps(out.seq) == pickle.dumps(seq)
    assert not hasattr(out, "circuits")

    # Multiple circuits:
    out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit]

    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert not hasattr(out, "circuit")

    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits([circuit, circuit]),
        "initial_logical_to_physicals": "[[], []]",
        "final_logical_to_physicals": "[[], []]",
        "state_jp": state_str,
        "readout_jp": state_str,
        "readout_qubits": "[4, 5, 6, 7]",
    }
    out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert isinstance(out.seq, qtrl.sequencer.Sequence)
    assert isinstance(out.seq._readout, qtrl.sequencer.Sequence)
    assert isinstance(out.seq._readout._readout, qtrl.sequence_utils.readout._ReadoutInfo)
    assert out.seq._readout._readout.sequence is out.seq._readout
    assert out.seq._readout._readout.qubits == [4, 5, 6, 7]
    assert out.seq._readout._readout.n_readouts == 2
    assert not hasattr(out, "circuit")


def test_read_json_qscout() -> None:
    circuit = qiskit.QuantumCircuit(1)
    circuit.h(0)

    jaqal_program = textwrap.dedent(
        """\
        register allqubits[1]

        prepare_all
        R allqubits[0] -1.5707963267948966 1.5707963267948966
        Rz allqubits[0] -3.141592653589793
        measure_all
        """
    )

    json_dict: dict[str, str | list[str]] = {
        "qiskit_circuits": qss.serialization.serialize_circuits(circuit),
        "initial_logical_to_physicals": json.dumps([[(0, 1)]]),
        "final_logical_to_physicals": json.dumps([[(0, 13)]]),
        "jaqal_programs": [jaqal_program],
    }

    out = qss.compiler_output.read_json_qscout(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert out.initial_logical_to_physical == {0: 1}
    assert out.final_logical_to_physical == {0: 13}
    assert out.jaqal_program == jaqal_program

    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits([circuit, circuit]),
        "initial_logical_to_physicals": json.dumps([[(0, 1)], [(0, 1)]]),
        "final_logical_to_physicals": json.dumps([[(0, 13)], [(0, 13)]]),
        "jaqal_programs": [jaqal_program, jaqal_program],
    }
    out = qss.compiler_output.read_json_qscout(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]
    assert out.initial_logical_to_physicals == [{0: 1}, {0: 1}]
    assert out.final_logical_to_physicals == [{0: 13}, {0: 13}]
    assert out.jaqal_programs == json_dict["jaqal_programs"]


def test_compiler_output_eq() -> None:
    circuit = qiskit.QuantumCircuit(1)
    circuit.h(0)
    co = qss.compiler_output.CompilerOutput(circuit, {0: 0}, {5: 0})
    assert co != 1

    circuit1 = qiskit.QuantumCircuit(1)
    circuit1.h(0)

    assert (
        qss.compiler_output.CompilerOutput([circuit, circuit1], [{0: 0}, {1: 1}], [{5: 0}, {4: 0}])
        != co
    )
