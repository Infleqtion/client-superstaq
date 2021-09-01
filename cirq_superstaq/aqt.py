import codecs
import importlib
import pickle
from typing import List, Optional, Union

import cirq

import cirq_superstaq

try:
    import qtrl.sequencer
except ModuleNotFoundError:
    pass


class AQTCompilerOutput:
    def __init__(
        self,
        circuits: Union[cirq.Circuit, List[cirq.Circuit]],
        seq: Optional["qtrl.sequencer.Sequence"] = None,
    ) -> None:
        if isinstance(circuits, cirq.Circuit):
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
    ):  # pragma: no cover, b/c qtrl is not open source so it is not in cirq-superstaq reqs
        state_str = json_dict["state_jp"]
        state = pickle.loads(codecs.decode(state_str.encode(), "base64"))

        seq = qtrl.sequencer.Sequence(n_elements=1)
        seq.__setstate__(state)
        seq.compile()

    resolvers = [cirq_superstaq.custom_gates.custom_resolver, *cirq.DEFAULT_RESOLVERS]
    compiled_circuits = [
        cirq.read_json(json_text=c, resolvers=resolvers) for c in json_dict["cirq_circuits"]
    ]
    if circuits_list:
        return AQTCompilerOutput(compiled_circuits, seq)

    return AQTCompilerOutput(compiled_circuits[0], seq)
