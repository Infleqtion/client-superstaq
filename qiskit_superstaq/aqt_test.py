import codecs
import importlib
import pickle
from unittest import mock

import pytest
import qiskit

from qiskit_superstaq import aqt


def test_aqt_compiler_output_repr() -> None:
    circuit = qiskit.QuantumCircuit(4)
    assert repr(aqt.AQTCompilerOutput(circuit)) == f"AQTCompilerOutput({circuit!r}, None)"

    circuits = [circuit, circuit]
    assert repr(aqt.AQTCompilerOutput(circuits)) == f"AQTCompilerOutput({circuits!r}, None)"


@mock.patch.dict("sys.modules", {"qtrl": None})
def test_read_json() -> None:
    importlib.reload(aqt)

    circuit = qiskit.QuantumCircuit(4)
    for i in range(4):
        circuit.h(i)
    state_str = codecs.encode(pickle.dumps({}), "base64").decode()

    json_dict = {
        "qasm_strs": [circuit.qasm()],
        "state_jp": state_str,
    }

    compiler_output = aqt.read_json(json_dict, circuits_list=False)
    assert compiler_output.circuit == circuit
    assert not hasattr(compiler_output, "circuits")

    compiler_output = aqt.read_json(json_dict, circuits_list=True)
    assert compiler_output.circuits == [circuit]
    assert not hasattr(compiler_output, "circuit")

    json_dict = {
        "qasm_strs": [circuit.qasm(), circuit.qasm()],
        "state_jp": state_str,
    }
    compiler_output = aqt.read_json(json_dict, circuits_list=True)
    assert compiler_output.circuits == [circuit, circuit]
    assert not hasattr(compiler_output, "circuit")


def test_read_json_with_qtrl() -> None:  # pragma: no cover, b/c test requires qtrl installation
    qtrl = pytest.importorskip("qtrl", reason="qtrl not installed")
    seq = qtrl.sequencer.Sequence(n_elements=1)

    circuit = qiskit.QuantumCircuit(4)
    for i in range(4):
        circuit.h(i)
    state_str = codecs.encode(pickle.dumps(seq.__getstate__()), "base64").decode()

    json_dict = {
        "qasm_strs": [circuit.qasm()],
        "state_jp": state_str,
    }
    compiler_output = aqt.read_json(json_dict, circuits_list=False)

    assert compiler_output.circuit == circuit
    assert not hasattr(compiler_output, "circuits")
    assert pickle.dumps(compiler_output.seq) == pickle.dumps(seq)

    compiler_output = aqt.read_json(json_dict, circuits_list=True)
    assert compiler_output.circuits == [circuit]
    assert not hasattr(compiler_output, "circuit")
    assert pickle.dumps(compiler_output.seq) == pickle.dumps(seq)

    json_dict = {
        "qasm_strs": [circuit.qasm(), circuit.qasm()],
        "state_jp": state_str,
    }
    compiler_output = aqt.read_json(json_dict, circuits_list=True)
    assert compiler_output.circuits == [circuit, circuit]
    assert not hasattr(compiler_output, "circuit")
    assert pickle.dumps(compiler_output.seq) == pickle.dumps(seq)
