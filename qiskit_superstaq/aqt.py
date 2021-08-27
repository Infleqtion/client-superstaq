import codecs
import importlib
import pickle
from dataclasses import dataclass
from typing import List, Optional, Union

import qiskit

try:
    import qtrl.sequencer
except ModuleNotFoundError:
    pass


@dataclass
class AQTCompilerOutput:
    circuit: qiskit.QuantumCircuit
    seq: Optional["qtrl.sequencer.Sequence"] = None


@dataclass
class AQTCompilerOutputMulti:
    circuits: List[qiskit.QuantumCircuit]
    seq: Optional["qtrl.sequencer.Sequence"] = None


def read_json(json_dict: dict) -> Union[AQTCompilerOutput, AQTCompilerOutputMulti]:
    """Reads out returned JSON from SuperstaQ API's AQT compilation endpoint.

    Args:
        json_dict: a JSON dictionary matching the format returned by /aqt_{multi_}compile endpoint
    Returns:
        a AQTCompilerOutput object with the compiled circuit or a AQTCompilerOutputMulti object
        with a list of compiled circuits. If qtrl is available locally, the returned object
        also stores the pulse sequence in the .seq attribute.
    """
    seq = None
    if importlib.util.find_spec(
        "qtrl"
    ):  # pragma: no cover, b/c qtrl is not open source so it is not in qiskit-superstaq reqs
        state_str = json_dict["state_jp"]
        state = pickle.loads(codecs.decode(state_str.encode(), "base64"))

        seq = qtrl.sequencer.Sequence(n_elements=1)
        seq.__setstate__(state)
        seq.compile()

    if "qasm_str" in json_dict:
        compiled_circuit = qiskit.QuantumCircuit.from_qasm_str(json_dict["qasm_str"])
        return AQTCompilerOutput(compiled_circuit, seq)

    compiled_circuits = [qiskit.QuantumCircuit.from_qasm_str(q) for q in json_dict["qasm_strs"]]
    return AQTCompilerOutputMulti(compiled_circuits, seq)
