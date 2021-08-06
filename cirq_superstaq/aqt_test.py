import cirq

from cirq_superstaq import aqt


def test_read_json() -> None:
    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(4)))
    json_dict = {"compiled_circuit": cirq.to_json(circuit)}
    assert aqt.read_json(json_dict) == aqt.AQTCompilerOutput(circuit)
