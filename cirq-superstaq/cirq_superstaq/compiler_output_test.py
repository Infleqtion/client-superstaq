from __future__ import annotations

import importlib
import pickle
import textwrap
from unittest import mock

import cirq
import general_superstaq as gss
import pytest
import qiskit
import qiskit_superstaq as qss

import cirq_superstaq as css


def test_active_qubit_indices() -> None:
    qubits = cirq.LineQubit.range(6)

    circuit = cirq.Circuit(
        cirq.X(qubits[5]),
        cirq.CZ(qubits[3], qubits[1]),
        css.barrier(*qubits),
        cirq.H(qubits[1]),
    )

    assert css.active_qubit_indices(circuit) == [1, 3, 5]

    with pytest.raises(ValueError, match="line qubits"):
        _ = css.active_qubit_indices(cirq.Circuit(cirq.X(cirq.GridQubit(1, 2))))


def test_measured_qubit_indices() -> None:
    # Create qubits with indices [0, 1, 2, 3, 5, 6]. No q4 to ensure that indices refer to
    # LineQubit arguments regardless of the number of qubits in the circuit
    q0, q1, q2, q3, _, q5, q6 = cirq.LineQubit.range(7)

    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.measure(q1),
        cirq.CX(q1, q2),
        cirq.measure(q6, q5),
        cirq.measure(q1, q3),  # (q1 was already measured)
        cirq.measure(q5, q1),  # (both were already measured)
    )
    assert css.measured_qubit_indices(circuit) == [1, 3, 5, 6]


def test_measured_qubit_indices_with_circuit_operations() -> None:
    """Check that measurements in CircuitOperations are mapped correctly."""
    # Create qubits with indices [0, 1, 2, 3, 5, 6]. No q4 to ensure that indices refer to
    # LineQubit arguments regardless of the number of qubits in the circuit
    q0, q1, q2, q3, _, q5, q6 = cirq.LineQubit.range(7)

    subcircuit = cirq.FrozenCircuit(cirq.X(q0), cirq.measure(q2, q3))
    assert css.measured_qubit_indices(subcircuit) == [2, 3]

    # Create a CircuitOperation with no qubit mapping (measurements don't move)
    subcircuit_op_no_map = cirq.CircuitOperation(subcircuit)
    assert css.measured_qubit_indices(subcircuit_op_no_map.mapped_circuit()) == [2, 3]

    # Create a CircuitOperation with a nontrivial map. Measurements (q2, q3) should land on (q5, q3)
    subcircuit_op_mapped = cirq.CircuitOperation(subcircuit).with_qubit_mapping({q1: q6, q2: q5})
    assert css.measured_qubit_indices(subcircuit_op_mapped.mapped_circuit()) == [3, 5]

    # Check that measured_qubit_indices() respects the qubit mapping
    circuit = cirq.Circuit(cirq.measure(q1))
    assert css.measured_qubit_indices(circuit) == [1]
    assert css.measured_qubit_indices(circuit + subcircuit_op_no_map) == [1, 2, 3]  # no mapping
    assert css.measured_qubit_indices(circuit + subcircuit_op_mapped) == [1, 3, 5]  # with mapping

    # Double-check that measurement indices are the same after unrolling subcircuit's qubit mapping
    unrolled_circuit = cirq.unroll_circuit_op(circuit + subcircuit_op_mapped, tags_to_check=None)
    assert css.measured_qubit_indices(unrolled_circuit) == [1, 3, 5]

    with pytest.raises(ValueError, match="line qubits"):
        _ = css.measured_qubit_indices(cirq.Circuit(cirq.measure(cirq.GridQubit(1, 2))))


def test_compiler_output_repr() -> None:
    circuit = cirq.Circuit()
    qubit_map: dict[cirq.Qid, cirq.Qid] = {}
    assert (
        repr(css.compiler_output.CompilerOutput(circuit, qubit_map, qubit_map))
        == f"CompilerOutput({circuit!r}, {{}}, {{}}, None, None, None)"
    )

    circuits = [circuit, circuit]
    assert (
        repr(css.compiler_output.CompilerOutput(circuits, [qubit_map], [qubit_map]))
        == f"CompilerOutput({circuits!r}, [{{}}], [{{}}], None, None, None)"
    )


def test_read_json() -> None:
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
    initial_logical_to_physical = {cirq.q(0): cirq.q(1)}
    final_logical_to_physical = {cirq.q(0): cirq.q(13)}

    json_dict = {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "initial_logical_to_physicals": cirq.to_json([list(initial_logical_to_physical.items())]),
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
    }

    out = css.compiler_output.read_json(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert out.initial_logical_to_physical == initial_logical_to_physical
    assert out.final_logical_to_physical == final_logical_to_physical
    assert not hasattr(out, "circuits")
    assert not hasattr(out, "initial_logical_to_physicals")
    assert not hasattr(out, "final_logical_to_physicals")

    out = css.compiler_output.read_json(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit]
    assert out.final_logical_to_physicals == [final_logical_to_physical]
    assert out.initial_logical_to_physicals == [initial_logical_to_physical]
    assert not hasattr(out, "circuit")
    assert not hasattr(out, "initial_logical_to_physical")
    assert not hasattr(out, "final_logical_to_physical")


def test_read_json_ibmq() -> None:
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
    initial_logical_to_physical = {cirq.q(0): cirq.q(1)}
    final_logical_to_physical = {cirq.q(0): cirq.q(13)}

    json_dict = {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "pulse_gate_circuits": qss.serialization.serialize_circuits([qiskit.QuantumCircuit()]),
        "initial_logical_to_physicals": cirq.to_json([list(initial_logical_to_physical.items())]),
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
    }

    out = css.compiler_output.read_json(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert out.pulse_gate_circuit == qiskit.QuantumCircuit()
    assert out.initial_logical_to_physical == initial_logical_to_physical
    assert out.final_logical_to_physical == final_logical_to_physical
    assert not hasattr(out, "circuits")
    assert not hasattr(out, "pulse_gate_circuits")
    assert not hasattr(out, "initial_logical_to_physicals")
    assert not hasattr(out, "final_logical_to_physicals")

    out = css.compiler_output.read_json(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit]
    assert out.pulse_gate_circuits == [qiskit.QuantumCircuit()]
    assert out.final_logical_to_physicals == [final_logical_to_physical]
    assert out.initial_logical_to_physicals == [initial_logical_to_physical]
    assert not hasattr(out, "circuit")
    assert not hasattr(out, "pulse_gate_circuit")
    assert not hasattr(out, "initial_logical_to_physical")
    assert not hasattr(out, "final_logical_to_physical")


def test_read_json_pulse_gate_circuits() -> None:
    qss = pytest.importorskip("qiskit_superstaq", reason="qiskit-superstaq is not installed")
    import qiskit  # noqa: PLC0415

    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))

    qc_pulse = qiskit.QuantumCircuit(2)
    qc_pulse.h(0)
    qc_pulse.cx(0, 1)

    json_dict = {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "initial_logical_to_physicals": "[[]]",
        "final_logical_to_physicals": "[[]]",
        "pulse_gate_circuits": qss.serialization.serialize_circuits(qc_pulse),
        "pulse_durations": [[10, 20]],
        "pulse_start_times": [[0, 10]],
    }

    out = css.compiler_output.read_json(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert out.pulse_gate_circuit == qc_pulse
    assert out.pulse_gate_circuit.duration == 30
    assert out.pulse_gate_circuit.op_start_times == [0, 10]

    json_dict = {
        "cirq_circuits": css.serialization.serialize_circuits([circuit, circuit]),
        "initial_logical_to_physicals": "[[], []]",
        "final_logical_to_physicals": "[[], []]",
        "pulse_gate_circuits": qss.serialization.serialize_circuits([qc_pulse, qc_pulse]),
        "pulse_durations": [[10, 20], [100, 200]],
        "pulse_start_times": [[0, 10], [0, 100]],
    }
    out = css.compiler_output.read_json(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]
    assert out.pulse_gate_circuits == [qc_pulse, qc_pulse]
    assert out.pulse_gate_circuits[0].duration == 30
    assert out.pulse_gate_circuits[1].duration == 300
    assert out.pulse_gate_circuits[1].op_start_times == [0, 100]

    with (
        mock.patch.dict("sys.modules", {"qiskit_superstaq": None}),
        pytest.warns(UserWarning, match="qiskit-superstaq is required"),
    ):
        out = css.compiler_output.read_json(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert out.pulse_gate_circuit is None

    json_dict["pulse_gate_circuits"] = "not-a-serialized-circuit"
    with pytest.warns(
        UserWarning,
        match="Your compiled pulse gate circuits could not be deserialized.",
    ):
        out = css.compiler_output.read_json(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]
    assert out.pulse_gate_circuits is None


@mock.patch.dict("sys.modules", {"qtrl": None})
def test_read_json_aqt() -> None:
    importlib.reload(css.compiler_output)

    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.H.on_each(*qubits), cirq.measure(*qubits))
    state_str = gss.serialization.serialize({})
    initial_logical_to_physical = {cirq.q(0): cirq.q(1)}
    final_logical_to_physical = {cirq.q(0): cirq.q(4)}

    json_dict = {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "state_jp": state_str,
        "initial_logical_to_physicals": cirq.to_json([list(initial_logical_to_physical.items())]),
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
    }

    with pytest.warns(UserWarning, match="deserialize compiled pulse sequences"):
        out = css.compiler_output.read_json_aqt(json_dict, circuits_is_list=False)

    assert out.circuit == circuit
    assert out.initial_logical_to_physical == initial_logical_to_physical
    assert out.final_logical_to_physical == final_logical_to_physical
    assert not hasattr(out, "circuits")
    assert not hasattr(out, "initial_logical_to_physicals")
    assert not hasattr(out, "final_logical_to_physicals")

    with pytest.warns(UserWarning, match="deserialize compiled pulse sequences"):
        out = css.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)

    assert out.circuits == [circuit]
    assert out.final_logical_to_physicals == [final_logical_to_physical]
    assert out.initial_logical_to_physicals == [initial_logical_to_physical]
    assert not hasattr(out, "circuit")
    assert not hasattr(out, "initial_logical_to_physical")
    assert not hasattr(out, "final_logical_to_physical")

    with pytest.warns(UserWarning, match="deserialize compiled pulse sequences"):
        out = css.compiler_output.read_json_aqt(json_dict, circuits_is_list=False)

    assert out.circuit == circuit
    assert out.seq is None

    # multiple circuits
    json_dict = {
        "cirq_circuits": css.serialization.serialize_circuits([circuit, circuit]),
        "state_jp": state_str,
        "initial_logical_to_physicals": cirq.to_json(
            2 * [list(initial_logical_to_physical.items())]
        ),
        "final_logical_to_physicals": cirq.to_json(2 * [list(final_logical_to_physical.items())]),
    }

    with pytest.warns(UserWarning, match="deserialize compiled pulse sequences"):
        out = css.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)

    assert out.circuits == [circuit, circuit]
    assert out.initial_logical_to_physicals == [
        initial_logical_to_physical,
        initial_logical_to_physical,
    ]
    assert out.final_logical_to_physicals == [final_logical_to_physical, final_logical_to_physical]
    assert not hasattr(out, "circuit")
    assert not hasattr(out, "initial_logical_to_physical")
    assert not hasattr(out, "final_logical_to_physical")

    # no sequence returned
    json_dict.pop("state_jp")
    out = css.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)
    assert out.seq is None


def test_read_json_with_qtrl() -> None:  # pragma: no cover, b/c test requires qtrl installation
    qtrl = pytest.importorskip("qtrl", reason="qtrl not installed")
    seq = qtrl.sequencer.Sequence(n_elements=1)
    initial_logical_to_physical = {cirq.q(0): cirq.q(1)}
    final_logical_to_physical = {cirq.q(0): cirq.q(4)}

    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(4)))
    state_str = gss.serialization.serialize(seq.__getstate__())
    json_dict = {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "state_jp": state_str,
        "initial_logical_to_physicals": cirq.to_json([list(initial_logical_to_physical.items())]),
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
    }

    out = css.compiler_output.read_json_aqt(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert isinstance(out.seq, qtrl.sequencer.Sequence)
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert not hasattr(out.seq, "_readout")
    assert not hasattr(out, "circuits")

    # Serialized readout attribute for aqt_zurich_qpu:
    json_dict["readout_jp"] = state_str
    json_dict["readout_qubits"] = "[4, 5, 6, 7]"
    out = css.compiler_output.read_json_aqt(json_dict, circuits_is_list=False)
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
    out = css.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit]
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert not hasattr(out, "circuit")

    json_dict = {
        "cirq_circuits": css.serialization.serialize_circuits([circuit, circuit]),
        "state_jp": state_str,
        "readout_jp": state_str,
        "readout_qubits": "[4, 5, 6, 7]",
        "initial_logical_to_physicals": cirq.to_json(
            2 * [list(initial_logical_to_physical.items())]
        ),
        "final_logical_to_physicals": cirq.to_json(2 * [list(final_logical_to_physical.items())]),
    }
    out = css.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)
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
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
    initial_logical_to_physical = {cirq.q(0): cirq.q(1)}
    final_logical_to_physical = {cirq.q(0): cirq.q(13)}

    jaqal_program = textwrap.dedent(
        """\
        register allqubits[1]
        prepare_all
        R allqubits[0] -1.5707963267948966 1.5707963267948966
        Rz allqubits[0] -3.141592653589793
        measure_all
        """
    )

    json_dict = {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "initial_logical_to_physicals": cirq.to_json([list(initial_logical_to_physical.items())]),
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
        "jaqal_programs": [jaqal_program],
    }

    out = css.compiler_output.read_json_qscout(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert out.initial_logical_to_physical == initial_logical_to_physical
    assert out.final_logical_to_physical == final_logical_to_physical
    assert out.jaqal_program == jaqal_program
    assert not hasattr(out, "initial_logical_to_physicals")
    assert not hasattr(out, "final_logical_to_physicals")
    assert not hasattr(out, "jaqal_programs")

    json_dict = {
        "cirq_circuits": css.serialization.serialize_circuits([circuit, circuit]),
        "initial_logical_to_physicals": cirq.to_json([list(initial_logical_to_physical.items())]),
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
        "jaqal_programs": [jaqal_program, jaqal_program],
    }
    out = css.compiler_output.read_json_qscout(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]
    assert out.final_logical_to_physicals == [final_logical_to_physical]
    assert out.initial_logical_to_physicals == [initial_logical_to_physical]
    assert out.jaqal_programs == json_dict["jaqal_programs"]
    assert not hasattr(out, "initial_logical_to_physical")
    assert not hasattr(out, "final_logical_to_physical")
    assert not hasattr(out, "jaqal_program")
