# Copyright 2026 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
"""Utilities for building fault-tolerant circuits with the Steane code."""

from __future__ import annotations

from collections.abc import Sequence

import cirq
import numpy as np

STEANE_CODE_SIZE = 7

_LOGICAL_ZERO_CODEWORDS = (
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 1, 1, 1),
    (0, 1, 1, 0, 0, 1, 1),
    (0, 1, 1, 1, 1, 0, 0),
    (1, 0, 1, 0, 1, 0, 1),
    (1, 0, 1, 1, 0, 1, 0),
    (1, 1, 0, 0, 1, 1, 0),
    (1, 1, 0, 1, 0, 0, 1),
)
_LOGICAL_ONE_CODEWORDS = tuple(
    tuple(1 - bit for bit in codeword) for codeword in _LOGICAL_ZERO_CODEWORDS
)
_SYNDROME_TO_INDEX = {
    (0, 0, 1): 0,
    (0, 1, 0): 1,
    (0, 1, 1): 2,
    (1, 0, 0): 3,
    (1, 0, 1): 4,
    (1, 1, 0): 5,
    (1, 1, 1): 6,
}


def _validate_block(qubits: Sequence[cirq.Qid]) -> None:
    if len(qubits) != STEANE_CODE_SIZE:
        raise ValueError(f"A Steane code block must contain {STEANE_CODE_SIZE} qubits.")
    if len(set(qubits)) != STEANE_CODE_SIZE:
        raise ValueError("A Steane code block cannot contain duplicate qubits.")


def transversal_h(qubits: Sequence[cirq.Qid]) -> list[cirq.Operation]:
    """Returns a transversal logical Hadamard operation."""
    _validate_block(qubits)
    return list(cirq.H.on_each(*qubits))


def transversal_x(qubits: Sequence[cirq.Qid]) -> list[cirq.Operation]:
    """Returns a transversal logical X operation."""
    _validate_block(qubits)
    return list(cirq.X.on_each(*qubits))


def transversal_cx(
    control_qubits: Sequence[cirq.Qid], target_qubits: Sequence[cirq.Qid]
) -> list[cirq.Operation]:
    """Returns pairwise CNOT operations between two equally sized code blocks."""
    if len(control_qubits) != len(target_qubits):
        raise ValueError("Control and target blocks must have the same size.")
    return [
        cirq.CX(control, target)
        for control, target in zip(control_qubits, target_qubits)
    ]


def encode(qubits: Sequence[cirq.Qid]) -> cirq.Circuit:
    """Returns a circuit that prepares the Steane-encoded state ``|0>_L``."""
    _validate_block(qubits)
    return cirq.Circuit(
        cirq.H.on_each(qubits[0], qubits[1], qubits[3]),
        cirq.CX(qubits[0], qubits[2]),
        cirq.CX(qubits[3], qubits[5]),
        cirq.CX(qubits[1], qubits[6]),
        cirq.CX(qubits[0], qubits[4]),
        cirq.CX(qubits[3], qubits[6]),
        cirq.CX(qubits[1], qubits[5]),
        cirq.CX(qubits[0], qubits[6]),
        cirq.CX(qubits[1], qubits[2]),
        cirq.CX(qubits[3], qubits[4]),
    )


def physical_measurements_to_logical_measurements(measurements: Sequence[int]) -> int:
    """Decodes a measured Steane codeword into a logical zero or one."""
    codeword = tuple(int(bit) for bit in measurements)
    if codeword in _LOGICAL_ZERO_CODEWORDS:
        return 0
    if codeword in _LOGICAL_ONE_CODEWORDS:
        return 1
    raise ValueError(
        f"The physical measurement is outside the Steane codespace: {codeword}"
    )


def syndrome(measurements: Sequence[int]) -> tuple[int, int, int]:
    """Calculates the three parity checks used to locate one bit-flip error."""
    if len(measurements) != STEANE_CODE_SIZE:
        raise ValueError(f"Expected {STEANE_CODE_SIZE} physical measurements.")
    return (
        int(measurements[3] ^ measurements[4] ^ measurements[5] ^ measurements[6]),
        int(measurements[1] ^ measurements[2] ^ measurements[5] ^ measurements[6]),
        int(measurements[0] ^ measurements[2] ^ measurements[4] ^ measurements[6]),
    )


def correct_error(
    measurements: Sequence[int],
    syndrome_measurements: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Returns a copy of a codeword with its syndrome-indicated bit corrected."""
    corrected = np.asarray(measurements, dtype=np.int8).copy()
    if corrected.shape != (STEANE_CODE_SIZE,):
        raise ValueError(f"Expected {STEANE_CODE_SIZE} physical measurements.")
    measured_syndrome = (
        syndrome(corrected) if syndrome_measurements is None else syndrome_measurements
    )
    error_index = _SYNDROME_TO_INDEX.get(tuple(map(int, measured_syndrome)))
    if error_index is not None:
        corrected[error_index] ^= 1
    return corrected


def qec_simulator(circuit: cirq.Circuit, repetitions: int = 1) -> cirq.ResultDict:
    """Simulates a Clifford circuit and decodes each seven-bit measurement."""
    result = cirq.CliffordSimulator().run(circuit, repetitions=repetitions)
    logical_measurements = {
        key: np.asarray(
            [[physical_measurements_to_logical_measurements(row)] for row in values],
            dtype=np.int8,
        )
        for key, values in result.measurements.items()
    }
    return cirq.ResultDict(params=result.params, measurements=logical_measurements)


def _resolve_and_correct(
    classical_data: cirq.ClassicalDataStore, key: cirq.MeasurementKey
) -> bool:
    # Cirq's condition API exposes recorded values but does not currently provide a public mutation
    # method. Correction is applied to a copy; only the decoded control value is needed here.
    measured = np.asarray(classical_data.records[key][0], dtype=np.int8)
    return bool(physical_measurements_to_logical_measurements(correct_error(measured)))


class SteaneCodeCondition(cirq.KeyCondition):
    """Classical condition that corrects and decodes a measured Steane code block."""

    def resolve(self, classical_data: cirq.ClassicalDataStore) -> bool:
        return _resolve_and_correct(classical_data, self.key)


def compile(circuit: cirq.AbstractCircuit) -> cirq.Circuit:
    """Compiles a logical circuit using Knill-style error-correcting teleportation."""
    logical_qubits = sorted(circuit.all_qubits())
    block_count = len(logical_qubits)
    physical_qubits = cirq.LineQubit.range(3 * STEANE_CODE_SIZE * block_count)

    def make_blocks(offset: int) -> dict[cirq.Qid, list[cirq.LineQubit]]:
        return {
            qubit: physical_qubits[
                offset
                + index * STEANE_CODE_SIZE : offset
                + (index + 1) * STEANE_CODE_SIZE
            ]
            for index, qubit in enumerate(logical_qubits)
        }

    data_blocks = make_blocks(0)
    ancilla_blocks = make_blocks(STEANE_CODE_SIZE * block_count)
    output_blocks = make_blocks(2 * STEANE_CODE_SIZE * block_count)
    compiled = cirq.Circuit.zip(
        *(
            encode(block)
            for blocks in (data_blocks, ancilla_blocks, output_blocks)
            for block in blocks.values()
        )
    )
    measurement_keys: set[str] = set()
    logical_measurements: list[tuple[str, tuple[cirq.Qid, ...]]] = []
    current_blocks = dict(data_blocks)
    alternate_blocks = dict(output_blocks)
    rounds = dict.fromkeys(logical_qubits, 0)

    for operation in circuit.all_operations():
        gate = operation.gate
        if isinstance(gate, cirq.MeasurementGate):
            key = cirq.measurement_key_name(operation)
            if key in measurement_keys:
                raise ValueError(f"Repeated measurement key {key!r} is not supported.")
            measurement_keys.add(key)
            logical_measurements.append((key, operation.qubits))
            continue
        if not (
            isinstance(gate, cirq.IdentityGate)
            or gate == cirq.X
            or gate == cirq.H
            or gate == cirq.CNOT
        ):
            raise ValueError(f"Unsupported logical operation: {operation!r}")

        # After the first round, the measured destination and ancilla blocks must be
        # reset and re-encoded before they can form the next encoded Bell pair.
        reused_qubits = [qubit for qubit in operation.qubits if rounds[qubit]]
        if reused_qubits:
            compiled.append(
                cirq.reset_each(
                    *(
                        physical
                        for qubit in reused_qubits
                        for physical in (
                            *alternate_blocks[qubit],
                            *ancilla_blocks[qubit],
                        )
                    )
                ),
                strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
            )
            compiled += cirq.Circuit.zip(
                *(
                    encode(block)
                    for qubit in reused_qubits
                    for block in (alternate_blocks[qubit], ancilla_blocks[qubit])
                )
            )

        if gate == cirq.X:
            compiled.append(
                transversal_x(current_blocks[operation.qubits[0]]),
                strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
            )
        elif gate == cirq.H:
            compiled.append(
                transversal_h(current_blocks[operation.qubits[0]]),
                strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
            )
        elif gate == cirq.CNOT:
            compiled.append(
                transversal_cx(
                    current_blocks[operation.qubits[0]],
                    current_blocks[operation.qubits[1]],
                ),
                strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
            )

        for qubit in operation.qubits:
            source = current_blocks[qubit]
            destination = alternate_blocks[qubit]
            round_index = rounds[qubit]
            data_key = f"__ft_data_{logical_qubits.index(qubit)}_{round_index}"
            ancilla_key = f"__ft_ancilla_{logical_qubits.index(qubit)}_{round_index}"

            compiled.append(
                transversal_h(ancilla_blocks[qubit]),
                strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
            )
            compiled.append(transversal_cx(ancilla_blocks[qubit], destination))
            compiled.append(transversal_cx(source, ancilla_blocks[qubit]))
            compiled.append(transversal_h(source))
            compiled.append(cirq.measure(*source, key=data_key))
            compiled.append(cirq.measure(*ancilla_blocks[qubit], key=ancilla_key))
            compiled.append(
                cirq.Z(physical).with_classical_controls(
                    SteaneCodeCondition(cirq.MeasurementKey(data_key))
                )
                for physical in destination
            )
            compiled.append(
                cirq.X(physical).with_classical_controls(
                    SteaneCodeCondition(cirq.MeasurementKey(ancilla_key))
                )
                for physical in destination
            )

            current_blocks[qubit], alternate_blocks[qubit] = destination, source
            rounds[qubit] += 1

    for key, measured_qubits in logical_measurements:
        compiled.append(
            cirq.measure(
                *(
                    physical
                    for qubit in measured_qubits
                    for physical in current_blocks[qubit]
                ),
                key=key,
            )
        )

    return compiled


class FTSimulator:
    """Simulates logical Cirq circuits using Steane-encoded physical qubits.

    The input circuit is expressed in terms of logical qubits. Each logical qubit uses
    three seven-qubit Steane blocks for error-correcting teleportation. Supported gates
    are applied transversally, and output-block measurements are decoded before being
    returned.

    Currently supported operations are identity, X, H, CNOT, and measurement.
    """

    def __init__(self, seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None) -> None:
        self._simulator = cirq.CliffordSimulator(seed=seed)

    @staticmethod
    def _measurement_qubit_counts(circuit: cirq.AbstractCircuit) -> dict[str, int]:
        measured_qubit_counts: dict[str, int] = {}
        for operation in circuit.all_operations():
            if isinstance(operation.gate, cirq.MeasurementGate):
                key = cirq.measurement_key_name(operation)
                measured_qubit_counts[key] = len(operation.qubits)
        return measured_qubit_counts

    def run(
        self,
        circuit: cirq.AbstractCircuit,
        *,
        repetitions: int = 1,
    ) -> cirq.ResultDict:
        """Compiles and simulates a logical circuit, returning logical measurements."""
        compiled = compile(circuit)
        measured_qubit_counts = self._measurement_qubit_counts(circuit)
        physical_result = self._simulator.run(compiled, repetitions=repetitions)
        logical_measurements: dict[str, np.ndarray] = {}

        for key, logical_qubit_count in measured_qubit_counts.items():
            physical_values = physical_result.measurements[key]
            decoded = np.empty((repetitions, logical_qubit_count), dtype=np.int8)
            for repetition, row in enumerate(physical_values):
                for logical_index in range(logical_qubit_count):
                    start = logical_index * STEANE_CODE_SIZE
                    decoded[
                        repetition, logical_index
                    ] = physical_measurements_to_logical_measurements(
                        row[start : start + STEANE_CODE_SIZE]
                    )
            logical_measurements[key] = decoded

        return cirq.ResultDict(
            params=physical_result.params, measurements=logical_measurements
        )


def generate_logical_circuit(
    logical_gate: cirq.OP_TREE,
    error_qubit: cirq.Qid,
    data_qubits: Sequence[cirq.Qid],
    ancilla0_qubits: Sequence[cirq.Qid],
    ancilla1_qubits: Sequence[cirq.Qid],
    *,
    error_probability: float = 0.5,
) -> cirq.Circuit:
    """Builds a Knill-style error-correcting teleportation circuit.

    ``logical_gate`` acts on ``data_qubits`` before its state is teleported to
    ``ancilla1_qubits``, which becomes the output block.
    """
    for block in (data_qubits, ancilla0_qubits, ancilla1_qubits):
        _validate_block(block)
    all_qubits = tuple(data_qubits) + tuple(ancilla0_qubits) + tuple(ancilla1_qubits)
    if len(set(all_qubits)) != 3 * STEANE_CODE_SIZE:
        raise ValueError("The data and ancilla blocks must be disjoint.")
    if error_qubit not in data_qubits:
        raise ValueError("error_qubit must belong to the data block.")

    return cirq.Circuit(
        encode(data_qubits),
        encode(ancilla0_qubits),
        encode(ancilla1_qubits),
        logical_gate,
        transversal_h(ancilla0_qubits),
        transversal_cx(ancilla0_qubits, ancilla1_qubits),
        transversal_cx(data_qubits, ancilla0_qubits),
        transversal_h(data_qubits),
        cirq.depolarize(p=error_probability).on(error_qubit),
        cirq.measure(data_qubits, key="ancilla0"),
        cirq.measure(ancilla0_qubits, key="ancilla1"),
        (
            cirq.Z(qubit).with_classical_controls(SteaneCodeCondition("ancilla0"))
            for qubit in ancilla1_qubits
        ),
        (
            cirq.X(qubit).with_classical_controls(SteaneCodeCondition("ancilla1"))
            for qubit in ancilla1_qubits
        ),
        cirq.measure(ancilla1_qubits, key="teleported_measurements"),
    )
