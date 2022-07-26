import importlib
import pickle
import textwrap
from unittest import mock

import general_superstaq as gss
import pytest
import qiskit

import qiskit_superstaq as qss


def test_compiler_output_repr() -> None:
    circuit = qiskit.QuantumCircuit(4)
    assert (
        repr(qss.compiler_output.CompilerOutput(circuit))
        == f"""CompilerOutput({circuit!r}, None, None, None)"""
    )

    circuits = [circuit, circuit]
    assert (
        repr(qss.compiler_output.CompilerOutput(circuits))
        == f"CompilerOutput({circuits!r}, None, None, None)"
    )


@mock.patch.dict("sys.modules", {"qtrl": None})
def test_read_json() -> None:
    importlib.reload(qss.compiler_output)

    circuit = qiskit.QuantumCircuit(4)
    for i in range(4):
        circuit.h(i)
    state_str = gss.converters.serialize({})
    pulse_lists_str = gss.converters.serialize([[[]]])

    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits(circuit),
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }

    out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert not hasattr(out, "circuits")

    out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit]
    assert not hasattr(out, "circuit")

    pulse_lists_str = gss.converters.serialize([[[]], [[]]])
    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits([circuit, circuit]),
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }
    out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]
    assert not hasattr(out, "circuit")

    json_dict = {"qiskit_circuits": qss.serialization.serialize_circuits(circuit)}

    out = qss.compiler_output.read_json_only_circuits(json_dict, circuits_is_list=False)
    assert out.circuit == circuit

    json_dict = {"qiskit_circuits": qss.serialization.serialize_circuits([circuit, circuit])}
    out = qss.compiler_output.read_json_only_circuits(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]


def test_read_json_with_qtrl() -> None:  # pragma: no cover, b/c test requires qtrl installation
    qtrl = pytest.importorskip("qtrl", reason="qtrl not installed")
    seq = qtrl.sequencer.Sequence(n_elements=1)

    circuit = qiskit.QuantumCircuit(4)
    for i in range(4):
        circuit.h(i)
    state_str = gss.converters.serialize(seq.__getstate__())
    pulse_lists_str = gss.converters.serialize([[[]]])
    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits(circuit),
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }

    out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert out.pulse_list == [[]]
    assert not hasattr(out, "circuits") and not hasattr(out, "pulse_lists")

    out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit]
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert out.pulse_lists == [[[]]]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")

    pulse_lists_str = gss.converters.serialize([[[]], [[]]])
    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits([circuit, circuit]),
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }
    out = qss.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert out.pulse_lists == [[[]], [[]]]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")


def test_read_json_with_qscout() -> None:
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

    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits(circuit),
        "jaqal_programs": [jaqal_program],
    }

    out = qss.compiler_output.read_json_qscout(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert out.jaqal_program == jaqal_program

    json_dict = {
        "qiskit_circuits": qss.serialization.serialize_circuits([circuit, circuit]),
        "jaqal_programs": [jaqal_program, jaqal_program],
    }
    out = qss.compiler_output.read_json_qscout(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]
    assert out.jaqal_programs == json_dict["jaqal_programs"]


def test_compiler_output_eq() -> None:
    circuit = qiskit.QuantumCircuit(1)
    circuit.h(0)
    co = qss.compiler_output.CompilerOutput(circuit)
    assert co != 1

    circuit1 = qiskit.QuantumCircuit(1)
    circuit1.h(0)

    assert qss.compiler_output.CompilerOutput([circuit, circuit1]) != co
