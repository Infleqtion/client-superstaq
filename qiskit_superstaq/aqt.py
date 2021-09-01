import codecs
import importlib
import pickle
from typing import List, Optional, Union

import qiskit

try:
    import qtrl.sequencer
except ModuleNotFoundError:
    pass


class AQTCompilerOutput:
    def __init__(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        seq: Optional["qtrl.sequencer.Sequence"] = None,
    ) -> None:
        if isinstance(circuits, qiskit.QuantumCircuit):
            self.circuit = circuits
        else:
            self.circuits = circuits
        self.seq = seq

    def __repr__(self) -> str:
        if hasattr(self, "circuit"):
            return f"AQTCompilerOutput({self.circuit!r}, {self.seq!r})"
        return f"AQTCompilerOutput({self.circuits!r}, {self.seq!r})"


def read_json(json_dict: dict, circuits_list: bool) -> AQTCompilerOutput:
    """Reads out returned JSON from SuperstaQ API's AQT compilation endpoint.

    Args:
        json_dict: a JSON dictionary matching the format returned by /aqt_compile endpoint
        circuits_list: bool flag that controls whether the returned object has a .circuits
            attribute (if True) or a .circuit attribute (False)
    Returns:
        a AQTCompilerOutput object with the compiled circuit(s). If qtrl is available locally,
        the returned object also stores the pulse sequence in the .seq attribute.
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

    compiled_circuits = [qiskit.QuantumCircuit.from_qasm_str(q) for q in json_dict["qasm_strs"]]
    if circuits_list:
        return AQTCompilerOutput(compiled_circuits, seq)

    return AQTCompilerOutput(compiled_circuits[0], seq)
