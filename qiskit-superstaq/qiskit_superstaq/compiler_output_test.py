# pylint: disable=missing-function-docstring,missing-class-docstring
import importlib
import json
import pickle
import textwrap
from typing import Dict, List, Union
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


def test_compiler_output_repr() -> None:
    circuit = qiskit.QuantumCircuit(4)
    assert (
        repr(qss.compiler_output.CompilerOutput(circuit, {0: 1}))
        == f"""CompilerOutput({circuit!r}, {{0: 1}}, None, None, None, None)"""
    )

    circuits = [circuit, circuit]
    assert (
        repr(qss.compiler_output.CompilerOutput(circuits, [{0: 1}, {1: 0}]))
        == f"CompilerOutput({circuits!r}, [{{0: 1}}, {{1: 0}}], None, None, None, None)"
    )


def test_read_json() -> None:
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    qc_pulse = qc.copy()
    qc_pulse.add_calibration("cx", [0, 1], qiskit.pulse.ScheduleBlock("foo"))

    pulse_sequence = mock.DEFAULT

    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": "[[]]",
        "pulse_gate_circuits": qss.serialization.serialize_circuits(qc_pulse),
        "pulses": gss.serialization.serialize([pulse_sequence]),
    }

    out = qss.compiler_output.read_json(json_dict, circuits_is_list=False)
    assert out.circuit == qc
    assert out.pulse_gate_circuit == qc_pulse

    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc, qc]),
        "final_logical_to_physicals": "[[], []]",
        "pulse_gate_circuits": qss.serialization.serialize_circuits([qc_pulse, qc_pulse]),
        "pulses": gss.serialization.serialize([pulse_sequence, pulse_sequence]),
    }
    out = qss.compiler_output.read_json(json_dict, circuits_is_list=True)
    assert out.circuits == [qc, qc]
    assert out.pulse_gate_circuits == [qc_pulse, qc_pulse]


@mock.patch.dict("sys.modules", {"qtrl": None})
def test_read_json_aqt() -> None:
    importlib.reload(qss.compiler_output)

    circuit = qiskit.QuantumCircuit(4)
    for i in range(4):
        circuit.h(i)

    state_str = gss.serialization.serialize({})
    pulse_lists_str = gss.serialization.serialize([[[]]])

    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits(circuit),
        "final_logical_to_physicals": "[[]]",
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
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
    pulse_lists_str = gss.serialization.serialize([[[]], [[]]])
    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits([circuit, circuit]),
        "final_logical_to_physicals": "[[], []]",
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }

    with pytest.warns(UserWarning, match="deserialize compiled pulse sequences"):
        out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)

    assert out.circuits == [circuit, circuit]
    assert not hasattr(out, "circuit")

    # no sequence returned
    json_dict.pop("state_jp")

    with pytest.warns(UserWarning, match="aqt_upload_configs"):
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
    pulse_lists_str = gss.serialization.serialize([[[]]])
    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits(circuit),
        "final_logical_to_physicals": "[[]]",
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }

    out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert isinstance(out.seq, qtrl.sequencer.Sequence)
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert out.pulse_list == [[]]
    assert not hasattr(out.seq, "_readout")
    assert not hasattr(out, "circuits") and not hasattr(out, "pulse_lists")

    # Serialized readout attribute for aqt_zurich_qpu:
    json_dict["readout_jp"] = state_str
    json_dict["readout_qubits"] = "[4, 5, 6, 7]"
    out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert out.pulse_list == [[]]
    assert isinstance(out.seq, qtrl.sequencer.Sequence)
    assert isinstance(out.seq._readout, qtrl.sequencer.Sequence)
    assert isinstance(out.seq._readout._readout, qtrl.sequence_utils.readout._ReadoutInfo)
    assert out.seq._readout._readout.sequence is out.seq._readout
    assert out.seq._readout._readout.qubits == [4, 5, 6, 7]
    assert out.seq._readout._readout.n_readouts == 1
    assert pickle.dumps(out.seq._readout) == pickle.dumps(out.seq) == pickle.dumps(seq)
    assert not hasattr(out, "circuits") and not hasattr(out, "pulse_lists")

    # Multiple circuits:
    out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit]

    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert out.pulse_lists == [[[]]]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")

    pulse_lists_str = gss.serialization.serialize([[[]], [[]]])
    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits([circuit, circuit]),
        "final_logical_to_physicals": "[[], []]",
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
        "readout_jp": state_str,
        "readout_qubits": "[4, 5, 6, 7]",
    }
    out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert out.pulse_lists == [[[]], [[]]]
    assert isinstance(out.seq, qtrl.sequencer.Sequence)
    assert isinstance(out.seq._readout, qtrl.sequencer.Sequence)
    assert isinstance(out.seq._readout._readout, qtrl.sequence_utils.readout._ReadoutInfo)
    assert out.seq._readout._readout.sequence is out.seq._readout
    assert out.seq._readout._readout.qubits == [4, 5, 6, 7]
    assert out.seq._readout._readout.n_readouts == 2
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")


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

    json_dict: Dict[str, Union[str, List[str]]] = {
        "qiskit_circuits": qss.serialization.serialize_circuits(circuit),
        "final_logical_to_physicals": json.dumps([[(0, 13)]]),
        "jaqal_programs": [jaqal_program],
    }

    out = qss.compiler_output.read_json_qscout(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert out.final_logical_to_physical == {0: 13}
    assert out.jaqal_program == jaqal_program

    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits([circuit, circuit]),
        "final_logical_to_physicals": json.dumps([[(0, 13)], [(0, 13)]]),
        "jaqal_programs": [jaqal_program, jaqal_program],
    }
    out = qss.compiler_output.read_json_qscout(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]
    assert out.final_logical_to_physicals == [{0: 13}, {0: 13}]
    assert out.jaqal_programs == json_dict["jaqal_programs"]


def test_compiler_output_eq() -> None:
    circuit = qiskit.QuantumCircuit(1)
    circuit.h(0)
    co = qss.compiler_output.CompilerOutput(circuit, {5: 0})
    assert co != 1

    circuit1 = qiskit.QuantumCircuit(1)
    circuit1.h(0)

    assert qss.compiler_output.CompilerOutput([circuit, circuit1], [{5: 0}, {4: 0}]) != co
