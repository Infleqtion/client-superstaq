import importlib
import pickle
from unittest import mock

import applications_superstaq
import pytest
import qiskit

from qiskit_superstaq import aqt


def test_aqt_compiler_output_repr() -> None:
    circuit = qiskit.QuantumCircuit(4)
    assert repr(aqt.AQTCompilerOutput(circuit)) == f"AQTCompilerOutput({circuit!r}, None, None)"

    circuits = [circuit, circuit]
    assert repr(aqt.AQTCompilerOutput(circuits)) == f"AQTCompilerOutput({circuits!r}, None, None)"


@mock.patch.dict("sys.modules", {"qtrl": None})
def test_read_json() -> None:
    importlib.reload(aqt)

    circuit = qiskit.QuantumCircuit(4)
    for i in range(4):
        circuit.h(i)
    state_str = applications_superstaq.converters.serialize({})
    pulse_lists_str = applications_superstaq.converters.serialize([[[]]])

    json_dict = {
        "qasm_strs": [circuit.qasm()],
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }

    out = aqt.read_json(json_dict, circuits_list=False)
    assert out.circuit == circuit
    assert not hasattr(out, "circuits")

    out = aqt.read_json(json_dict, circuits_list=True)
    assert out.circuits == [circuit]
    assert not hasattr(out, "circuit")

    pulse_lists_str = applications_superstaq.converters.serialize([[[]], [[]]])
    json_dict = {
        "qasm_strs": [circuit.qasm(), circuit.qasm()],
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }
    out = aqt.read_json(json_dict, circuits_list=True)
    assert out.circuits == [circuit, circuit]
    assert not hasattr(out, "circuit")


def test_read_json_with_qtrl() -> None:  # pragma: no cover, b/c test requires qtrl installation
    qtrl = pytest.importorskip("qtrl", reason="qtrl not installed")
    seq = qtrl.sequencer.Sequence(n_elements=1)

    circuit = qiskit.QuantumCircuit(4)
    for i in range(4):
        circuit.h(i)
    state_str = applications_superstaq.converters.serialize(seq.__getstate__())
    pulse_lists_str = applications_superstaq.converters.serialize([[[]]])
    json_dict = {
        "qasm_strs": [circuit.qasm()],
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }

    out = aqt.read_json(json_dict, circuits_list=False)
    assert out.circuit == circuit
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert out.pulse_list == [[]]
    assert not hasattr(out, "circuits") and not hasattr(out, "pulse_lists")

    out = aqt.read_json(json_dict, circuits_list=True)
    assert out.circuits == [circuit]
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert out.pulse_lists == [[[]]]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")

    pulse_lists_str = applications_superstaq.converters.serialize([[[]], [[]]])
    json_dict = {
        "qasm_strs": [circuit.qasm(), circuit.qasm()],
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }
    out = aqt.read_json(json_dict, circuits_list=True)
    assert out.circuits == [circuit, circuit]
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert out.pulse_lists == [[[]], [[]]]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")
