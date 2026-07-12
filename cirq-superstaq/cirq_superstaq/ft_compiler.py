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

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import cirq
import numpy as np
import qualtran

STEANE_CODE_SIZE = 7
REED_MULLER_15_CODE_SIZE = 15

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


@dataclass(frozen=True)
class ReedMuller15Code:
    """The ``[[15, 1, 3]]`` code used by Gottesman Protocol 13.4."""

    x_stabilizer_supports: tuple[tuple[int, ...], ...] = (
        tuple(range(8)),
        (0, 1, 2, 3, 8, 9, 10, 11),
        (0, 1, 4, 5, 8, 9, 12, 13),
        (0, 2, 4, 6, 8, 10, 12, 14),
    )
    z_stabilizer_supports: tuple[tuple[int, ...], ...] = (
        (0, 1, 2, 3),
        (0, 1, 4, 5),
        (0, 2, 4, 6),
        (0, 1, 8, 9),
        (0, 2, 8, 10),
        (0, 4, 8, 12),
        tuple(range(8)),
        (0, 1, 2, 3, 8, 9, 10, 11),
        (0, 1, 4, 5, 8, 9, 12, 13),
        (0, 2, 4, 6, 8, 10, 12, 14),
    )
    logical_pivot: int = 8
    stabilizer_pivots: tuple[int, ...] = (0, 1, 2, 4)
    systematic_rows: tuple[tuple[int, tuple[int, ...]], ...] = (
        (8, (9, 10, 11, 12, 13, 14)),
        (0, (3, 5, 6, 8, 11, 13, 14)),
        (1, (3, 5, 7, 8, 10, 12, 14)),
        (2, (3, 6, 7, 8, 9, 12, 13)),
        (4, (5, 6, 7, 8, 9, 10, 11)),
    )

    def phase_error_syndrome(self, z_errors: Sequence[int]) -> tuple[int, ...]:
        """Returns the four X-stabilizer checks detecting input magic-state Z errors."""
        if len(z_errors) != REED_MULLER_15_CODE_SIZE:
            raise ValueError("The 15-qubit Reed-Muller code requires 15 error bits.")
        return tuple(
            sum(int(z_errors[index]) for index in support) % 2
            for support in self.x_stabilizer_supports
        )

    def accepts_phase_errors(self, z_errors: Sequence[int]) -> bool:
        """Returns whether the distillation projection accepts this error pattern."""
        return not any(self.phase_error_syndrome(z_errors))

    def output_has_phase_error(self, z_errors: Sequence[int]) -> bool:
        """Returns the decoded logical-Z error for an accepted input pattern."""
        if not self.accepts_phase_errors(z_errors):
            raise ValueError("Rejected Reed-Muller error patterns have no output state.")
        return bool(sum(map(int, z_errors)) % 2)


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


def transversal_s(qubits: Sequence[cirq.Qid]) -> list[cirq.Operation]:
    """Returns the Steane logical S operation.

    With this encoder convention, physical ``S**-1`` on every qubit implements
    logical S on the encoded block.
    """
    _validate_block(qubits)
    return [(cirq.S**-1)(qubit) for qubit in qubits]


def transversal_cx(
    control_qubits: Sequence[cirq.Qid], target_qubits: Sequence[cirq.Qid]
) -> list[cirq.Operation]:
    """Returns pairwise CNOT operations between two equally sized code blocks."""
    if len(control_qubits) != len(target_qubits):
        raise ValueError("Control and target blocks must have the same size.")
    return [cirq.CX(control, target) for control, target in zip(control_qubits, target_qubits)]


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
    raise ValueError(f"The physical measurement is outside the Steane codespace: {codeword}")


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


def _resolve_and_correct(classical_data: cirq.ClassicalDataStore, key: cirq.MeasurementKey) -> bool:
    # Cirq's condition API exposes recorded values but does not currently provide a public mutation
    # method. Correction is applied to a copy; only the decoded control value is needed here.
    measured = np.asarray(classical_data.records[key][0], dtype=np.int8)
    return bool(physical_measurements_to_logical_measurements(correct_error(measured)))


class SteaneCodeCondition(cirq.KeyCondition):
    """Classical condition that corrects and decodes a measured Steane code block."""

    def resolve(self, classical_data: cirq.ClassicalDataStore) -> bool:
        return _resolve_and_correct(classical_data, self.key)


@dataclass(frozen=True)
class SteanePauliFrameUpdate:
    """Physical and logical Pauli corrections decoded from a Bell measurement."""

    physical_x_index: int | None
    physical_z_index: int | None
    logical_x: bool
    logical_z: bool


class SteaneBellMeasurementDecoder:
    """Decodes the two physical measurement blocks produced by Knill EC."""

    @staticmethod
    def decode(
        data_measurements: Sequence[int], bell_a_measurements: Sequence[int]
    ) -> SteanePauliFrameUpdate:
        """Returns the physical and logical updates for the teleported block."""
        data_syndrome = syndrome(data_measurements)
        bell_a_syndrome = syndrome(bell_a_measurements)
        corrected_data = correct_error(data_measurements, data_syndrome)
        corrected_bell_a = correct_error(bell_a_measurements, bell_a_syndrome)
        return SteanePauliFrameUpdate(
            physical_x_index=_SYNDROME_TO_INDEX.get(bell_a_syndrome),
            physical_z_index=_SYNDROME_TO_INDEX.get(data_syndrome),
            logical_x=bool(physical_measurements_to_logical_measurements(corrected_bell_a)),
            logical_z=bool(physical_measurements_to_logical_measurements(corrected_data)),
        )


@dataclass(frozen=True)
class SteaneSyndromeCondition(cirq.KeyCondition):
    """Classical condition selecting one physical correction from a Steane syndrome."""

    error_index: int = -1

    def resolve(self, classical_data: cirq.ClassicalDataStore) -> bool:
        measured = np.asarray(classical_data.records[self.key][0], dtype=np.int8)
        return _SYNDROME_TO_INDEX.get(syndrome(measured)) == self.error_index


def steane_zero_verification_passed(measurements: Sequence[int]) -> bool:
    """Returns whether a Steane-zero verification measurement should be accepted."""
    try:
        return physical_measurements_to_logical_measurements(measurements) == 0
    except ValueError:
        return False


@dataclass(frozen=True)
class VerifiedSteaneZero(qualtran.GateWithRegisters):
    """A postselected Qualtran Bloq that prepares ``|0>_L`` for the Steane code.

    ``candidate`` and ``verifier`` must both start in ``|0>``. Each block is
    encoded independently, after which a transversal CNOT compares the candidate
    with the verifier and the verifier is measured in the Z basis. The candidate
    is a verified output only when :func:`steane_zero_verification_passed` returns
    true for the measurement under ``verification_key``. A failed attempt must be
    discarded; retry and routing policy intentionally belong to a higher-level
    resource factory.

    This is the optimized distance-three Steane preparation described in
    Gottesman, section 13.1.2, figure 13.3.
    """

    verification_key: str = "steane_zero_verification"

    @property
    def signature(self) -> Any:
        return qualtran.Signature.build(
            candidate=STEANE_CODE_SIZE,
            verifier=STEANE_CODE_SIZE,
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        candidate: np.ndarray,
        verifier: np.ndarray,
    ) -> cirq.OP_TREE:
        del context
        candidate_qubits = tuple(candidate.ravel())
        verifier_qubits = tuple(verifier.ravel())

        yield cirq.Circuit.zip(encode(candidate_qubits), encode(verifier_qubits))
        yield transversal_cx(candidate_qubits, verifier_qubits)
        yield cirq.measure(*verifier_qubits, key=self.verification_key)


@dataclass(frozen=True)
class VerifiedSteanePlus(qualtran.GateWithRegisters):
    """A postselected ``|+>_L`` preparation composed from ``VerifiedSteaneZero``.

    The verification measurement and acceptance rule are inherited from the
    underlying zero-state preparation. The retained candidate is transformed to
    ``|+>_L`` by the Steane code's transversal logical Hadamard.
    """

    verification_key: str = "steane_plus_verification"

    @property
    def signature(self) -> Any:
        return qualtran.Signature.build(
            candidate=STEANE_CODE_SIZE,
            verifier=STEANE_CODE_SIZE,
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        candidate: np.ndarray,
        verifier: np.ndarray,
    ) -> cirq.OP_TREE:
        del context
        candidate_qubits = tuple(candidate.ravel())

        yield VerifiedSteaneZero(self.verification_key).on_registers(
            candidate=candidate,
            verifier=verifier,
        )
        yield transversal_h(candidate_qubits)


def steane_bell_pair_verification_passed(
    plus_measurements: Sequence[int], zero_measurements: Sequence[int]
) -> bool:
    """Returns whether both preparations underlying a Steane Bell pair passed."""
    return steane_zero_verification_passed(plus_measurements) and steane_zero_verification_passed(
        zero_measurements
    )


@dataclass(frozen=True)
class VerifiedSteaneBellPair(qualtran.GateWithRegisters):
    """A postselected encoded Bell pair composed from verified Steane states.

    The ``bell_a`` block is prepared as ``|+>_L`` and controls a transversal CNOT
    into ``bell_b``, which is prepared as ``|0>_L``. The output blocks form
    ``(|00>_L + |11>_L) / sqrt(2)`` only when the measurements under both
    verification keys pass :func:`steane_bell_pair_verification_passed`.
    """

    plus_verification_key: str = "steane_bell_plus_verification"
    zero_verification_key: str = "steane_bell_zero_verification"

    @property
    def signature(self) -> Any:
        return qualtran.Signature.build(
            bell_a=STEANE_CODE_SIZE,
            bell_a_verifier=STEANE_CODE_SIZE,
            bell_b=STEANE_CODE_SIZE,
            bell_b_verifier=STEANE_CODE_SIZE,
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        bell_a: np.ndarray,
        bell_a_verifier: np.ndarray,
        bell_b: np.ndarray,
        bell_b_verifier: np.ndarray,
    ) -> cirq.OP_TREE:
        del context
        bell_a_qubits = tuple(bell_a.ravel())
        bell_b_qubits = tuple(bell_b.ravel())

        yield VerifiedSteanePlus(self.plus_verification_key).on_registers(
            candidate=bell_a,
            verifier=bell_a_verifier,
        )
        yield VerifiedSteaneZero(self.zero_verification_key).on_registers(
            candidate=bell_b,
            verifier=bell_b_verifier,
        )
        yield transversal_cx(bell_a_qubits, bell_b_qubits)


def _steane_magic_state_vector() -> np.ndarray:
    zero = np.zeros(2**STEANE_CODE_SIZE, dtype=np.complex128)
    one = np.zeros_like(zero)
    for codeword in _LOGICAL_ZERO_CODEWORDS:
        zero[cirq.big_endian_bits_to_int(codeword)] = 1 / np.sqrt(8)
    for codeword in _LOGICAL_ONE_CODEWORDS:
        one[cirq.big_endian_bits_to_int(codeword)] = 1 / np.sqrt(8)
    return (zero + np.exp(1j * np.pi / 4) * one) / np.sqrt(2)


class EncodedMagicStateSource(Protocol):
    """Composable source interface consumed by non-Clifford injection gadgets."""

    def preparation_bloq(self, key_prefix: str = "magic_state") -> qualtran.GateWithRegisters:
        """Returns a Bloq that prepares one encoded ``T|+>`` resource."""
        ...

    def accepts(self, measurements: Mapping[str, Sequence[int]], key_prefix: str) -> bool:
        """Returns whether a postselected preparation attempt succeeded."""
        ...


@dataclass(frozen=True)
class TrustedSteaneMagicState(qualtran.GateWithRegisters):
    """Prepares an exact encoded Steane magic state from a trusted primitive."""

    @property
    def signature(self) -> Any:
        return qualtran.Signature.build(magic=STEANE_CODE_SIZE)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        magic: np.ndarray,
    ) -> cirq.OP_TREE:
        del context
        yield cirq.StatePreparationChannel(
            _steane_magic_state_vector(), name="TrustedSteaneMagicState"
        ).on(*tuple(magic.ravel()))


@dataclass(frozen=True)
class TrustedSteaneMagicStateSource:
    """Functional source implementation replaceable by verification or distillation."""

    def preparation_bloq(self, key_prefix: str = "magic_state") -> qualtran.GateWithRegisters:
        del key_prefix
        return TrustedSteaneMagicState()

    def accepts(self, measurements: Mapping[str, Sequence[int]], key_prefix: str) -> bool:
        del measurements, key_prefix
        return True


@dataclass(frozen=True)
class TGateInjection(qualtran.GateWithRegisters):
    """Consumes an encoded magic state to apply logical T to a Steane block."""

    measurement_key: str = "t_injection_magic"

    @property
    def signature(self) -> Any:
        return qualtran.Signature.build(data=STEANE_CODE_SIZE, magic=STEANE_CODE_SIZE)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        data: np.ndarray,
        magic: np.ndarray,
    ) -> cirq.OP_TREE:
        del context
        data_qubits = tuple(data.ravel())
        magic_qubits = tuple(magic.ravel())
        yield transversal_cx(data_qubits, magic_qubits)
        yield cirq.measure(*magic_qubits, key=self.measurement_key)
        yield (
            operation.with_classical_controls(
                SteaneCodeCondition(cirq.MeasurementKey(self.measurement_key))
            )
            for operation in transversal_s(data_qubits)
        )


def reed_muller_distillation_passed(
    measurements: Mapping[str, Sequence[int]], key_prefix: str
) -> bool:
    """Returns whether all preparation and Reed-Muller syndrome checks passed."""
    check_count = len(ReedMuller15Code().x_stabilizer_supports) + len(
        ReedMuller15Code().z_stabilizer_supports
    )
    return all(
        steane_zero_verification_passed(measurements[f"{key_prefix}_verify_{index}"])
        and steane_zero_verification_passed(measurements[f"{key_prefix}_syndrome_{index}"])
        for index in range(check_count)
    )


@dataclass(frozen=True)
class FifteenToOneMagicStateDistillation(qualtran.GateWithRegisters):
    """Gottesman's encoded 15-to-1 magic-state distillation protocol.

    ``magic`` contains fifteen Steane-encoded noisy magic states. On acceptance,
    block 8 contains the distilled state; all other blocks and both ancillas are
    consumed. Rejected attempts must be discarded and restarted.
    """

    key_prefix: str = "fifteen_to_one"

    @property
    def signature(self) -> Any:
        return qualtran.Signature.build(
            magic=REED_MULLER_15_CODE_SIZE * STEANE_CODE_SIZE,
            ancilla=STEANE_CODE_SIZE,
            verifier=STEANE_CODE_SIZE,
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        magic: np.ndarray,
        ancilla: np.ndarray,
        verifier: np.ndarray,
    ) -> cirq.OP_TREE:
        del context
        blocks = tuple(
            tuple(block)
            for block in np.asarray(magic).reshape(REED_MULLER_15_CODE_SIZE, STEANE_CODE_SIZE)
        )
        ancilla_block = tuple(ancilla.ravel())
        verifier_block = tuple(verifier.ravel())
        code = ReedMuller15Code()
        checks = tuple(("x", support) for support in code.x_stabilizer_supports) + tuple(
            ("z", support) for support in code.z_stabilizer_supports
        )

        for check_index, (basis, support) in enumerate(checks):
            yield cirq.reset_each(*ancilla_block, *verifier_block)
            verification_key = f"{self.key_prefix}_verify_{check_index}"
            if basis == "x":
                yield VerifiedSteanePlus(verification_key).on_registers(
                    candidate=np.asarray(ancilla_block),
                    verifier=np.asarray(verifier_block),
                )
                for block_index in support:
                    yield transversal_cx(ancilla_block, blocks[block_index])
                yield transversal_h(ancilla_block)
            else:
                yield VerifiedSteaneZero(verification_key).on_registers(
                    candidate=np.asarray(ancilla_block),
                    verifier=np.asarray(verifier_block),
                )
                for block_index in support:
                    yield transversal_cx(blocks[block_index], ancilla_block)
            yield cirq.measure(
                *ancilla_block,
                key=f"{self.key_prefix}_syndrome_{check_index}",
            )

        encoder_cnots = [
            (pivot, target) for pivot, targets in code.systematic_rows for target in targets
        ]
        for control, target in reversed(encoder_cnots):
            yield transversal_cx(blocks[control], blocks[target])
        for pivot in code.stabilizer_pivots:
            yield transversal_h(blocks[pivot])
        yield transversal_s(blocks[code.logical_pivot])


@dataclass(frozen=True)
class KnillCorrectedMagicState(qualtran.GateWithRegisters):
    """Prepares a raw encoded magic state and refreshes it through Knill EC."""

    raw_source: EncodedMagicStateSource
    key_prefix: str = "corrected_magic"

    @property
    def signature(self) -> Any:
        return qualtran.Signature.build(magic=STEANE_CODE_SIZE)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        magic: np.ndarray,
    ) -> cirq.OP_TREE:
        allocated = context.qubit_manager.qalloc(4 * STEANE_CODE_SIZE)
        data, bell_a, bell_a_verifier, bell_b_verifier = (
            np.asarray(allocated[start : start + STEANE_CODE_SIZE])
            for start in range(0, len(allocated), STEANE_CODE_SIZE)
        )
        protocol = VerifiedKnillEC(f"{self.key_prefix}_ec")
        yield self.raw_source.preparation_bloq(f"{self.key_prefix}_raw").on_registers(magic=data)
        yield protocol.bell_pair.on_registers(
            bell_a=bell_a,
            bell_a_verifier=bell_a_verifier,
            bell_b=magic,
            bell_b_verifier=bell_b_verifier,
        )
        yield protocol.teleportation.on_registers(
            data=data,
            bell_a=bell_a,
            bell_b=magic,
        )
        context.qubit_manager.qfree(allocated)


@dataclass(frozen=True)
class KnillCorrectedMagicStateSource:
    """Source adapter applying verified Knill EC to every raw magic candidate."""

    raw_source: EncodedMagicStateSource = TrustedSteaneMagicStateSource()

    def preparation_bloq(self, key_prefix: str = "magic_state") -> qualtran.GateWithRegisters:
        return KnillCorrectedMagicState(self.raw_source, key_prefix)

    def accepts(self, measurements: Mapping[str, Sequence[int]], key_prefix: str) -> bool:
        protocol = VerifiedKnillEC(f"{key_prefix}_ec")
        return self.raw_source.accepts(
            measurements, f"{key_prefix}_raw"
        ) and protocol.verification_passed(measurements)


@dataclass(frozen=True)
class DistilledSteaneMagicState(qualtran.GateWithRegisters):
    """Prepares one postselected distilled state from fifteen raw source Bloqs."""

    raw_source: EncodedMagicStateSource
    key_prefix: str = "distilled_magic"

    @property
    def signature(self) -> Any:
        return qualtran.Signature.build(magic=STEANE_CODE_SIZE)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        magic: np.ndarray,
    ) -> cirq.OP_TREE:
        allocated = context.qubit_manager.qalloc(
            16 * STEANE_CODE_SIZE
        )  # Fourteen inputs plus syndrome ancilla and verifier.
        allocated_blocks = [
            tuple(allocated[start : start + STEANE_CODE_SIZE])
            for start in range(0, len(allocated), STEANE_CODE_SIZE)
        ]
        magic_blocks = allocated_blocks[:14]
        magic_blocks.insert(ReedMuller15Code().logical_pivot, tuple(magic.ravel()))
        ancilla_block, verifier_block = allocated_blocks[14:]

        for input_index, block in enumerate(magic_blocks):
            yield self.raw_source.preparation_bloq(
                f"{self.key_prefix}_raw_{input_index}"
            ).on_registers(magic=np.asarray(block))
        yield FifteenToOneMagicStateDistillation(self.key_prefix).on_registers(
            magic=np.asarray(magic_blocks).reshape(-1),
            ancilla=np.asarray(ancilla_block),
            verifier=np.asarray(verifier_block),
        )
        context.qubit_manager.qfree(allocated)


@dataclass(frozen=True)
class DistilledSteaneMagicStateSource:
    """Postselected 15-to-1 source implementing Gottesman Protocol 13.4."""

    raw_source: EncodedMagicStateSource = KnillCorrectedMagicStateSource()

    def preparation_bloq(self, key_prefix: str = "magic_state") -> qualtran.GateWithRegisters:
        return DistilledSteaneMagicState(self.raw_source, key_prefix)

    def accepts(self, measurements: Mapping[str, Sequence[int]], key_prefix: str) -> bool:
        return all(
            self.raw_source.accepts(measurements, f"{key_prefix}_raw_{index}")
            for index in range(REED_MULLER_15_CODE_SIZE)
        ) and reed_muller_distillation_passed(measurements, key_prefix)


@dataclass(frozen=True)
class KnillTeleportation(qualtran.GateWithRegisters):
    """Teleports an encoded Steane block through an accepted logical Bell pair.

    ``bell_a`` and ``bell_b`` must be the two retained blocks from an accepted
    :class:`VerifiedSteaneBellPair` attempt. This Bloq deliberately performs no
    resource preparation: keeping verification away from live ``data`` makes the
    acceptance and retry boundary explicit. ``data`` and ``bell_a`` are consumed
    by measurement, and the teleported logical state is left in ``bell_b``.
    """

    measurement_key_prefix: str = "knill_teleportation"

    @property
    def signature(self) -> Any:
        return qualtran.Signature.build(
            data=STEANE_CODE_SIZE,
            bell_a=STEANE_CODE_SIZE,
            bell_b=STEANE_CODE_SIZE,
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        data: np.ndarray,
        bell_a: np.ndarray,
        bell_b: np.ndarray,
    ) -> cirq.OP_TREE:
        del context
        data_qubits = tuple(data.ravel())
        bell_a_qubits = tuple(bell_a.ravel())
        bell_b_qubits = tuple(bell_b.ravel())
        data_key = f"{self.measurement_key_prefix}_data"
        bell_a_key = f"{self.measurement_key_prefix}_bell_a"

        yield transversal_cx(data_qubits, bell_a_qubits)
        yield transversal_h(data_qubits)
        yield cirq.measure(*data_qubits, key=data_key)
        yield cirq.measure(*bell_a_qubits, key=bell_a_key)
        yield (
            cirq.Z(qubit).with_classical_controls(
                SteaneSyndromeCondition(cirq.MeasurementKey(data_key), index)
            )
            for index, qubit in enumerate(bell_b_qubits)
        )
        yield (
            cirq.X(qubit).with_classical_controls(
                SteaneSyndromeCondition(cirq.MeasurementKey(bell_a_key), index)
            )
            for index, qubit in enumerate(bell_b_qubits)
        )
        yield (
            cirq.Z(qubit).with_classical_controls(
                SteaneCodeCondition(cirq.MeasurementKey(data_key))
            )
            for qubit in bell_b_qubits
        )
        yield (
            cirq.X(qubit).with_classical_controls(
                SteaneCodeCondition(cirq.MeasurementKey(bell_a_key))
            )
            for qubit in bell_b_qubits
        )


@dataclass(frozen=True)
class VerifiedKnillEC:
    """Coordinates verified resource preparation and Knill teleportation.

    This is intentionally not a single Bloq. Resource verification is
    postselected and must finish successfully before the Bell pair is allowed to
    interact with live data. ``bell_pair`` and ``teleportation`` expose the two
    independently schedulable Bloq stages; a resource factory may retry or prepare
    multiple copies between them.
    """

    key_prefix: str = "verified_knill_ec"

    @property
    def plus_verification_key(self) -> str:
        return f"{self.key_prefix}_plus_verification"

    @property
    def zero_verification_key(self) -> str:
        return f"{self.key_prefix}_zero_verification"

    @property
    def bell_pair(self) -> VerifiedSteaneBellPair:
        """Returns the offline, postselected Bell-resource preparation Bloq."""
        return VerifiedSteaneBellPair(
            plus_verification_key=self.plus_verification_key,
            zero_verification_key=self.zero_verification_key,
        )

    @property
    def teleportation(self) -> KnillTeleportation:
        """Returns the Bloq that consumes a previously accepted Bell pair."""
        return KnillTeleportation(measurement_key_prefix=f"{self.key_prefix}_teleportation")

    def verification_passed(self, measurements: Mapping[str, Sequence[int]]) -> bool:
        """Returns whether one resource-preparation result is safe to consume."""
        return steane_bell_pair_verification_passed(
            measurements[self.plus_verification_key],
            measurements[self.zero_verification_key],
        )


@dataclass(frozen=True)
class VerifiedSteaneBellPairFactory:
    """Creates a bounded batch of verified Bell-pair preparation attempts.

    The factory owns only classical orchestration: physical block allocation,
    parallel scheduling, and routing the selected blocks remain responsibilities
    of the compiler or runtime. Each attempt has independent measurement keys so
    that preparation may happen without interacting with live data.
    """

    attempts: int
    key_prefix: str = "steane_bell_pair"

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("A Bell-pair factory must make at least one attempt.")

    def protocol(self, attempt_index: int) -> VerifiedKnillEC:
        """Returns the independently keyed protocol for one factory attempt."""
        if not 0 <= attempt_index < self.attempts:
            raise IndexError(f"Bell-pair attempt index out of range: {attempt_index}")
        return VerifiedKnillEC(key_prefix=f"{self.key_prefix}_{attempt_index}")

    @property
    def preparations(self) -> tuple[VerifiedSteaneBellPair, ...]:
        """Returns the Bell-preparation Bloqs in this bounded batch."""
        return tuple(self.protocol(index).bell_pair for index in range(self.attempts))

    def select(self, measurements: Mapping[str, Sequence[int]]) -> int:
        """Returns the index of the first accepted Bell pair.

        Raises:
            RuntimeError: If every preparation attempt was rejected.
        """
        for index in range(self.attempts):
            if self.protocol(index).verification_passed(measurements):
                return index
        raise RuntimeError("No verified Steane Bell-pair preparation was accepted.")


@dataclass(frozen=True)
class SteaneBellPairBlocks:
    """Physical blocks allocated to one verified Bell-pair preparation attempt."""

    bell_a: tuple[cirq.Qid, ...]
    bell_a_verifier: tuple[cirq.Qid, ...]
    bell_b: tuple[cirq.Qid, ...]
    bell_b_verifier: tuple[cirq.Qid, ...]


@dataclass(frozen=True)
class VerifiedExecutionResource:
    """Verified Bell resources allocated to one logical block in one round."""

    logical_qubit_index: int
    factory: VerifiedSteaneBellPairFactory
    bell_pair_blocks: tuple[SteaneBellPairBlocks, ...]


@dataclass(frozen=True)
class VerifiedExecutionRound:
    """One logical operation and resources for each participating block."""

    logical_gate: cirq.Gate
    logical_qubit_indices: tuple[int, ...]
    resources: tuple[VerifiedExecutionResource, ...]
    moment_index: int
    magic_block: tuple[cirq.Qid, ...] | None = None


@dataclass(frozen=True)
class VerifiedExecutionPlan:
    """A staged verified compilation for an arbitrary-depth Clifford circuit.

    The resource preparation circuit must execute first while preserving the
    unmeasured Bell blocks. Once its classical results are available,
    :meth:`build_continuation` selects accepted attempts and chains the live data
    locations independently for every logical block.
    """

    logical_qubits: tuple[cirq.Qid, ...]
    data_blocks: tuple[tuple[cirq.Qid, ...], ...]
    rounds: tuple[VerifiedExecutionRound, ...]
    measured_logical_qubit_indices: tuple[int, ...]
    measurement_key: str
    magic_state_source: EncodedMagicStateSource | None = None

    @property
    def data(self) -> tuple[cirq.Qid, ...]:
        """Returns the initial data block for a one-logical-qubit plan."""
        if len(self.data_blocks) != 1:
            raise ValueError("A multi-qubit execution plan has more than one data block.")
        return self.data_blocks[0]

    @property
    def resource_preparation_circuit(self) -> cirq.Circuit:
        """Returns independently schedulable, offline Bell-pair preparations."""
        operations = []
        for round_ in self.rounds:
            if round_.magic_block is not None:
                assert self.magic_state_source is not None
                operations.append(
                    self.magic_state_source.preparation_bloq(
                        f"verified_magic_{round_.moment_index}"
                    ).on_registers(magic=np.asarray(round_.magic_block))
                )
            for resource in round_.resources:
                for preparation, blocks in zip(
                    resource.factory.preparations, resource.bell_pair_blocks
                ):
                    operations.append(
                        preparation.on_registers(
                            bell_a=np.asarray(blocks.bell_a),
                            bell_a_verifier=np.asarray(blocks.bell_a_verifier),
                            bell_b=np.asarray(blocks.bell_b),
                            bell_b_verifier=np.asarray(blocks.bell_b_verifier),
                        )
                    )
        return cirq.Circuit(operations)

    def build_continuation(
        self, verification_measurements: Mapping[str, Sequence[int]]
    ) -> cirq.Circuit:
        """Builds the data-touching stage using accepted pairs for every block."""
        circuit = cirq.Circuit.zip(*(encode(block) for block in self.data_blocks))
        current_blocks = list(self.data_blocks)

        for round_ in self.rounds:
            if round_.logical_gate == cirq.X:
                circuit.append(transversal_x(current_blocks[round_.logical_qubit_indices[0]]))
            elif round_.logical_gate == cirq.H:
                circuit.append(transversal_h(current_blocks[round_.logical_qubit_indices[0]]))
            elif round_.logical_gate == cirq.CNOT:
                control_index, target_index = round_.logical_qubit_indices
                circuit.append(
                    transversal_cx(current_blocks[control_index], current_blocks[target_index])
                )
            elif round_.logical_gate == cirq.T:
                assert round_.magic_block is not None
                assert self.magic_state_source is not None
                magic_key_prefix = f"verified_magic_{round_.moment_index}"
                if not self.magic_state_source.accepts(verification_measurements, magic_key_prefix):
                    raise RuntimeError("Encoded magic-state distillation was rejected.")
                circuit.append(
                    TGateInjection(
                        measurement_key=f"verified_t_injection_{round_.moment_index}"
                    ).on_registers(
                        data=np.asarray(current_blocks[round_.logical_qubit_indices[0]]),
                        magic=np.asarray(round_.magic_block),
                    )
                )
            elif not isinstance(round_.logical_gate, cirq.IdentityGate):
                raise ValueError(f"Unsupported verified logical gate: {round_.logical_gate!r}")

            for resource in round_.resources:
                selected_index = resource.factory.select(verification_measurements)
                selected_blocks = resource.bell_pair_blocks[selected_index]
                protocol = resource.factory.protocol(selected_index)
                circuit.append(
                    protocol.teleportation.on_registers(
                        data=np.asarray(current_blocks[resource.logical_qubit_index]),
                        bell_a=np.asarray(selected_blocks.bell_a),
                        bell_b=np.asarray(selected_blocks.bell_b),
                    )
                )
                current_blocks[resource.logical_qubit_index] = selected_blocks.bell_b

        circuit.append(
            cirq.measure(
                *(
                    physical_qubit
                    for logical_index in self.measured_logical_qubit_indices
                    for physical_qubit in current_blocks[logical_index]
                ),
                key=self.measurement_key,
            )
        )
        return circuit


@dataclass(frozen=True)
class LocalFTModule:
    """A fixed-capacity module in a local fault-tolerant architecture."""

    module_id: str
    logical_capacity: int = 1
    bell_pair_slots: int = 1
    bell_pair_attempt_capacity: int = 1
    magic_state_slots: int = 0

    def __post_init__(self) -> None:
        if (
            self.logical_capacity < 1
            or self.bell_pair_slots < 1
            or self.bell_pair_attempt_capacity < 1
            or self.magic_state_slots < 0
        ):
            raise ValueError("Local module capacities must be positive.")


@dataclass(frozen=True)
class LocalFTArchitecture:
    """Modules and direct communication links available to the compiler."""

    modules: tuple[LocalFTModule, ...]
    links: frozenset[tuple[str, str]]

    def __post_init__(self) -> None:
        module_ids = [module.module_id for module in self.modules]
        if len(set(module_ids)) != len(module_ids):
            raise ValueError("Local module identifiers must be unique.")
        known_ids = set(module_ids)
        for first, second in self.links:
            if first == second or first not in known_ids or second not in known_ids:
                raise ValueError(f"Invalid local architecture link: {(first, second)!r}")

    def module(self, module_id: str) -> LocalFTModule:
        try:
            return next(module for module in self.modules if module.module_id == module_id)
        except StopIteration as ex:
            raise ValueError(f"Unknown local module: {module_id!r}") from ex

    def are_neighbors(self, first: str, second: str) -> bool:
        return first == second or (first, second) in self.links or (second, first) in self.links

    def shortest_path(self, first: str, second: str) -> tuple[str, ...]:
        """Returns a shortest path of local links between two modules."""
        self.module(first)
        self.module(second)
        queue = deque([(first, (first,))])
        visited = {first}
        while queue:
            current, path = queue.popleft()
            if current == second:
                return path
            neighbors = {
                right if left == current else left
                for left, right in self.links
                if left == current or right == current
            }
            for neighbor in sorted(neighbors - visited):
                visited.add(neighbor)
                queue.append((neighbor, path + (neighbor,)))
        raise ValueError(f"No local route connects modules {first!r} and {second!r}.")


@dataclass(frozen=True)
class LocalResourceAssignment:
    """A reusable module-owned Bell-resource slot assigned to one logical block."""

    logical_qubit_index: int
    module_id: str
    slot_index: int


@dataclass(frozen=True)
class LocalScheduledRound:
    """Local routing and bounded resource assignments for one logical round."""

    round_index: int
    moment_index: int
    migrations: tuple[tuple[int, tuple[str, ...]], ...]
    resources: tuple[LocalResourceAssignment, ...]


@dataclass(frozen=True)
class LocalVerifiedExecutionPlan:
    """A verified execution plan with fixed logical-module ownership."""

    verified_plan: VerifiedExecutionPlan
    architecture: LocalFTArchitecture
    logical_module_ids: tuple[str, ...]
    scheduled_rounds: tuple[LocalScheduledRound, ...]

    def module_for_logical_index(self, logical_index: int) -> LocalFTModule:
        return self.architecture.module(self.logical_module_ids[logical_index])

    @property
    def physical_qubit_bound(self) -> int:
        """Returns the fixed module capacity, independent of program depth."""
        return sum(
            STEANE_CODE_SIZE * module.logical_capacity
            + 4 * STEANE_CODE_SIZE * module.bell_pair_slots * module.bell_pair_attempt_capacity
            + STEANE_CODE_SIZE * module.magic_state_slots
            for module in self.architecture.modules
        )


def compile_local_verified(
    circuit: cirq.AbstractCircuit,
    architecture: LocalFTArchitecture,
    placement: Mapping[cirq.Qid, str],
    *,
    preparation_attempts: int = 1,
    magic_state_source: EncodedMagicStateSource | None = None,
) -> LocalVerifiedExecutionPlan:
    """Compiles a verified program while enforcing module capacity and locality."""
    plan = compile_verified(
        circuit,
        preparation_attempts=preparation_attempts,
        magic_state_source=magic_state_source,
    )
    if set(placement) != set(plan.logical_qubits):
        raise ValueError("Placement must assign every logical qubit exactly once.")
    logical_module_ids = [placement[qubit] for qubit in plan.logical_qubits]
    occupancy = {module.module_id: 0 for module in architecture.modules}
    for module_id in logical_module_ids:
        module = architecture.module(module_id)
        occupancy[module_id] += 1
        if occupancy[module_id] > module.logical_capacity:
            raise ValueError(f"Logical capacity exceeded for module {module_id!r}.")
        if preparation_attempts > module.bell_pair_attempt_capacity:
            raise ValueError(f"Bell-pair attempt capacity exceeded for module {module_id!r}.")

    scheduled_rounds = []
    slots_used: dict[tuple[int, str], int] = {}
    for round_index, round_ in enumerate(plan.rounds):
        migrations = []
        if round_.logical_gate == cirq.T:
            module_id = logical_module_ids[round_.logical_qubit_indices[0]]
            if architecture.module(module_id).magic_state_slots < 1:
                raise ValueError(f"Module {module_id!r} has no encoded magic-state slot.")
        if round_.logical_gate == cirq.CNOT:
            control_index, target_index = round_.logical_qubit_indices
            control_module = logical_module_ids[control_index]
            target_module = logical_module_ids[target_index]
            path = architecture.shortest_path(control_module, target_module)
            if len(path) > 2:
                destination = path[-2]
                if occupancy[destination] >= architecture.module(destination).logical_capacity:
                    raise ValueError(
                        f"Logical capacity exceeded while routing into {destination!r}."
                    )
                occupancy[control_module] -= 1
                occupancy[destination] += 1
                logical_module_ids[control_index] = destination
                migrations.append((control_index, path[:-1]))

        resource_assignments = []
        for logical_index in round_.logical_qubit_indices:
            module_id = logical_module_ids[logical_index]
            slot_key = (round_.moment_index, module_id)
            slot_index = slots_used.get(slot_key, 0)
            module = architecture.module(module_id)
            if slot_index >= module.bell_pair_slots:
                raise ValueError(
                    f"Bell-pair slot capacity exceeded in module {module_id!r} "
                    f"during moment {round_.moment_index}."
                )
            slots_used[slot_key] = slot_index + 1
            resource_assignments.append(
                LocalResourceAssignment(logical_index, module_id, slot_index)
            )
        scheduled_rounds.append(
            LocalScheduledRound(
                round_index=round_index,
                moment_index=round_.moment_index,
                migrations=tuple(migrations),
                resources=tuple(resource_assignments),
            )
        )

    return LocalVerifiedExecutionPlan(
        plan,
        architecture,
        tuple(logical_module_ids),
        tuple(scheduled_rounds),
    )


def compile_verified(
    circuit: cirq.AbstractCircuit,
    *,
    preparation_attempts: int = 3,
    magic_state_source: EncodedMagicStateSource | None = None,
) -> VerifiedExecutionPlan:
    """Compiles a multi-qubit Clifford program into two verified stages."""
    logical_qubits = tuple(sorted(circuit.all_qubits()))
    logical_qubit_indices = {qubit: index for index, qubit in enumerate(logical_qubits)}

    logical_operations = []
    measurement_keys = []
    for moment_index, moment in enumerate(circuit):
        for operation in moment.operations:
            if isinstance(operation.gate, cirq.MeasurementGate):
                measurement_keys.append(cirq.measurement_key_name(operation))
            else:
                logical_operations.append((moment_index, operation))
    operations = tuple(circuit.all_operations())
    if (
        not logical_operations
        or len(measurement_keys) != 1
        or not isinstance(operations[-1].gate, cirq.MeasurementGate)
    ):
        raise ValueError(
            "Verified compilation requires at least one logical gate and one final measurement."
        )
    for _, logical_operation in logical_operations:
        logical_gate = logical_operation.gate
        if logical_gate is None or not (
            logical_gate == cirq.X
            or logical_gate == cirq.H
            or logical_gate == cirq.CNOT
            or logical_gate == cirq.T
            or isinstance(logical_gate, cirq.IdentityGate)
        ):
            raise ValueError(f"Unsupported verified logical operation: {logical_operation!r}")
        if logical_gate == cirq.T and magic_state_source is None:
            raise ValueError("Logical T requires an encoded magic-state source.")

    measurement_operation = operations[-1]
    measured_indices = tuple(logical_qubit_indices[qubit] for qubit in measurement_operation.qubits)
    data_blocks = tuple(
        tuple(
            cirq.LineQubit.range(
                logical_index * STEANE_CODE_SIZE,
                (logical_index + 1) * STEANE_CODE_SIZE,
            )
        )
        for logical_index in range(len(logical_qubits))
    )
    rounds = []
    next_physical_qubit = len(logical_qubits) * STEANE_CODE_SIZE
    for round_index, (moment_index, logical_operation) in enumerate(logical_operations):
        participating_indices = tuple(
            logical_qubit_indices[qubit] for qubit in logical_operation.qubits
        )
        resources = []
        for logical_index in participating_indices:
            allocated_attempts = []
            for _ in range(preparation_attempts):
                start = next_physical_qubit
                next_physical_qubit += 4 * STEANE_CODE_SIZE
                blocks = tuple(
                    tuple(cirq.LineQubit.range(start + offset, start + offset + STEANE_CODE_SIZE))
                    for offset in range(0, 4 * STEANE_CODE_SIZE, STEANE_CODE_SIZE)
                )
                allocated_attempts.append(SteaneBellPairBlocks(*blocks))
            resources.append(
                VerifiedExecutionResource(
                    logical_qubit_index=logical_index,
                    bell_pair_blocks=tuple(allocated_attempts),
                    factory=VerifiedSteaneBellPairFactory(
                        attempts=preparation_attempts,
                        key_prefix=(
                            f"verified_compile_round_{round_index}_logical_{logical_index}"
                        ),
                    ),
                )
            )
        magic_block = None
        if logical_operation.gate == cirq.T:
            magic_block = tuple(
                cirq.LineQubit.range(next_physical_qubit, next_physical_qubit + STEANE_CODE_SIZE)
            )
            next_physical_qubit += STEANE_CODE_SIZE
        rounds.append(
            VerifiedExecutionRound(
                logical_gate=logical_operation.gate,
                logical_qubit_indices=participating_indices,
                resources=tuple(resources),
                moment_index=moment_index,
                magic_block=magic_block,
            )
        )

    return VerifiedExecutionPlan(
        logical_qubits=logical_qubits,
        data_blocks=data_blocks,
        rounds=tuple(rounds),
        measured_logical_qubit_indices=measured_indices,
        measurement_key=measurement_keys[0],
        magic_state_source=magic_state_source,
    )


_FT_BLOQ_TYPES = (
    VerifiedSteaneZero,
    VerifiedSteanePlus,
    VerifiedSteaneBellPair,
    TrustedSteaneMagicState,
    TGateInjection,
    FifteenToOneMagicStateDistillation,
    KnillCorrectedMagicState,
    DistilledSteaneMagicState,
    KnillTeleportation,
)


def _lower_ft_operation(operation: cirq.Operation) -> list[cirq.Operation]:
    if not isinstance(operation.gate, _FT_BLOQ_TYPES):
        return [operation]
    decomposition = cirq.decompose_once(operation.gate, qubits=operation.qubits)
    return [
        lowered
        for child in cirq.flatten_op_tree(decomposition)
        for lowered in _lower_ft_operation(child)
    ]


def lower_ft_circuit(circuit: cirq.AbstractCircuit) -> cirq.Circuit:
    """Recursively lowers the fault-tolerant Qualtran Bloqs into Cirq operations."""
    lowered = cirq.Circuit()
    for moment in circuit:
        lowered.append(
            (
                operation
                for original in moment.operations
                for operation in _lower_ft_operation(original)
            )
        )
    return lowered


class VerifiedFTSimulator:
    """Executes verified staged plans while preserving accepted resource states."""

    def __init__(
        self,
        *,
        preparation_attempts: int = 3,
        max_preparation_batches: int = 10,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ) -> None:
        if max_preparation_batches < 1:
            raise ValueError("max_preparation_batches must be positive.")
        self.preparation_attempts = preparation_attempts
        self.max_preparation_batches = max_preparation_batches
        self._simulator = cirq.CliffordSimulator(seed=seed)

    def _run_once(self, plan: VerifiedExecutionPlan) -> np.ndarray:
        preparation = lower_ft_circuit(plan.resource_preparation_circuit)
        all_qubits = sorted(
            preparation.all_qubits() | {qubit for block in plan.data_blocks for qubit in block}
        )

        for _ in range(self.max_preparation_batches):
            preparation_result = self._simulator.simulate(
                preparation,
                qubit_order=all_qubits,
            )
            try:
                continuation = plan.build_continuation(preparation_result.measurements)
            except RuntimeError:
                continue

            lowered_continuation = lower_ft_circuit(continuation)
            continuation_result = self._simulator.simulate(
                lowered_continuation,
                qubit_order=all_qubits,
                initial_state=preparation_result._final_simulator_state,
            )
            physical_measurements = continuation_result.measurements[plan.measurement_key]
            return np.asarray(
                [
                    physical_measurements[index * STEANE_CODE_SIZE : (index + 1) * STEANE_CODE_SIZE]
                    for index in range(len(plan.measured_logical_qubit_indices))
                ]
            )

        raise RuntimeError("Verified resource preparation exhausted its retry budget.")

    def run(
        self,
        circuit: cirq.AbstractCircuit,
        *,
        repetitions: int = 1,
    ) -> cirq.ResultDict:
        """Compiles and executes a verified logical Clifford circuit."""
        plan = compile_verified(circuit, preparation_attempts=self.preparation_attempts)
        decoded = np.empty((repetitions, len(plan.measured_logical_qubit_indices)), dtype=np.int8)
        for repetition in range(repetitions):
            physical_blocks = self._run_once(plan)
            for logical_index, physical_measurements in enumerate(physical_blocks):
                decoded[repetition, logical_index] = physical_measurements_to_logical_measurements(
                    physical_measurements
                )
        return cirq.ResultDict(
            params=cirq.ParamResolver({}),
            measurements={plan.measurement_key: decoded},
        )


@dataclass(frozen=True)
class LocalDecoderEvent:
    """A Bell-measurement decode performed within one module."""

    module_id: str
    data_key: str
    bell_a_key: str


@dataclass(frozen=True)
class LocalClassicalMessage:
    """A nearest-neighbor Pauli-frame message sent after routed teleportation."""

    source_module: str
    destination_module: str
    data_key: str
    bell_a_key: str


class LocalVerifiedFTSimulator:
    """Executes verified operations using reset-and-reused local physical pools."""

    def __init__(
        self,
        architecture: LocalFTArchitecture,
        placement: Mapping[cirq.Qid, str],
        *,
        preparation_attempts: int = 1,
        max_preparation_batches: int = 10,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ) -> None:
        self.architecture = architecture
        self.placement = dict(placement)
        self.preparation_attempts = preparation_attempts
        self.max_preparation_batches = max_preparation_batches
        self._simulator = cirq.CliffordSimulator(seed=seed)
        self.last_plan: LocalVerifiedExecutionPlan | None = None
        self.decoder_events: list[LocalDecoderEvent] = []
        self.classical_messages: list[LocalClassicalMessage] = []
        self.executed_reset_count = 0
        self.physical_qubits_used = 0
        self._key_counter = 0

    def _new_key_prefix(self, label: str) -> str:
        prefix = f"local_{label}_{self._key_counter}"
        self._key_counter += 1
        return prefix

    def _allocate_module_blocks(self) -> dict[str, tuple[tuple[cirq.Qid, ...], ...]]:
        blocks = {}
        next_qubit = 0
        for module in self.architecture.modules:
            block_count = (
                module.logical_capacity
                + 4 * module.bell_pair_slots * module.bell_pair_attempt_capacity
            )
            module_blocks = []
            for _ in range(block_count):
                module_blocks.append(
                    tuple(cirq.LineQubit.range(next_qubit, next_qubit + STEANE_CODE_SIZE))
                )
                next_qubit += STEANE_CODE_SIZE
            blocks[module.module_id] = tuple(module_blocks)
        self.physical_qubits_used = next_qubit
        return blocks

    def _simulate_stage(
        self,
        circuit: cirq.Circuit,
        all_qubits: Sequence[cirq.Qid],
        state: Any = None,
    ) -> Any:
        return self._simulator.simulate(
            lower_ft_circuit(circuit),
            qubit_order=all_qubits,
            initial_state=state,
        )

    @staticmethod
    def _free_blocks(
        module_id: str,
        module_blocks: Mapping[str, tuple[tuple[cirq.Qid, ...], ...]],
        current_blocks: Mapping[int, tuple[cirq.Qid, ...]],
        current_modules: Mapping[int, str],
    ) -> list[tuple[cirq.Qid, ...]]:
        occupied = {
            current_blocks[index] for index, owner in current_modules.items() if owner == module_id
        }
        return [block for block in module_blocks[module_id] if block not in occupied]

    def _prepare_bell_pair(
        self,
        source_module: str,
        destination_module: str,
        module_blocks: Mapping[str, tuple[tuple[cirq.Qid, ...], ...]],
        current_blocks: Mapping[int, tuple[cirq.Qid, ...]],
        current_modules: Mapping[int, str],
        all_qubits: Sequence[cirq.Qid],
        state: Any,
    ) -> tuple[Any, tuple[cirq.Qid, ...], tuple[cirq.Qid, ...], VerifiedKnillEC]:
        if source_module == destination_module:
            free = self._free_blocks(source_module, module_blocks, current_blocks, current_modules)
            if len(free) < 4:
                raise RuntimeError(f"Module {source_module!r} has no free Bell-resource slot.")
            bell_a, bell_a_verifier, bell_b, bell_b_verifier = free[:4]
        else:
            source_free = self._free_blocks(
                source_module, module_blocks, current_blocks, current_modules
            )
            destination_free = self._free_blocks(
                destination_module, module_blocks, current_blocks, current_modules
            )
            if len(source_free) < 2 or len(destination_free) < 2:
                raise RuntimeError("A routed Bell link requires two free blocks at each endpoint.")
            bell_a, bell_a_verifier = source_free[:2]
            bell_b, bell_b_verifier = destination_free[:2]

        resource_qubits = bell_a + bell_a_verifier + bell_b + bell_b_verifier
        for _ in range(self.max_preparation_batches):
            prefix = self._new_key_prefix("resource")
            protocol = VerifiedKnillEC(prefix)
            preparation = cirq.Circuit(
                cirq.reset_each(*resource_qubits),
                protocol.bell_pair.on_registers(
                    bell_a=np.asarray(bell_a),
                    bell_a_verifier=np.asarray(bell_a_verifier),
                    bell_b=np.asarray(bell_b),
                    bell_b_verifier=np.asarray(bell_b_verifier),
                ),
            )
            self.executed_reset_count += len(resource_qubits)
            result = self._simulate_stage(preparation, all_qubits, state)
            if protocol.verification_passed(result.measurements):
                return result._final_simulator_state, bell_a, bell_b, protocol
            state = result._final_simulator_state
        raise RuntimeError("Verified local resource preparation exhausted its retry budget.")

    def _teleport(
        self,
        logical_index: int,
        source_module: str,
        destination_module: str,
        module_blocks: Mapping[str, tuple[tuple[cirq.Qid, ...], ...]],
        current_blocks: dict[int, tuple[cirq.Qid, ...]],
        current_modules: dict[int, str],
        all_qubits: Sequence[cirq.Qid],
        state: Any,
    ) -> Any:
        state, bell_a, bell_b, protocol = self._prepare_bell_pair(
            source_module,
            destination_module,
            module_blocks,
            current_blocks,
            current_modules,
            all_qubits,
            state,
        )
        teleportation = protocol.teleportation.on_registers(
            data=np.asarray(current_blocks[logical_index]),
            bell_a=np.asarray(bell_a),
            bell_b=np.asarray(bell_b),
        )
        result = self._simulate_stage(cirq.Circuit(teleportation), all_qubits, state)
        data_key = f"{protocol.teleportation.measurement_key_prefix}_data"
        bell_a_key = f"{protocol.teleportation.measurement_key_prefix}_bell_a"
        self.decoder_events.append(LocalDecoderEvent(source_module, data_key, bell_a_key))
        if source_module != destination_module:
            if not self.architecture.are_neighbors(source_module, destination_module):
                raise RuntimeError("Classical Pauli-frame messages may only cross one local link.")
            self.classical_messages.append(
                LocalClassicalMessage(source_module, destination_module, data_key, bell_a_key)
            )
        current_blocks[logical_index] = bell_b
        current_modules[logical_index] = destination_module
        return result._final_simulator_state

    def _run_once(self, circuit: cirq.AbstractCircuit) -> np.ndarray:
        assert self.last_plan is not None
        plan = self.last_plan
        module_blocks = self._allocate_module_blocks()
        all_qubits = sorted(
            qubit for blocks in module_blocks.values() for block in blocks for qubit in block
        )
        current_modules = {
            index: self.placement[qubit]
            for index, qubit in enumerate(plan.verified_plan.logical_qubits)
        }
        used_data_slots = {module.module_id: 0 for module in self.architecture.modules}
        current_blocks = {}
        for logical_index, module_id in current_modules.items():
            slot = used_data_slots[module_id]
            current_blocks[logical_index] = module_blocks[module_id][slot]
            used_data_slots[module_id] += 1

        initialization = cirq.Circuit.zip(*(encode(block) for block in current_blocks.values()))
        result = self._simulate_stage(initialization, all_qubits)
        state = result._final_simulator_state
        logical_operations = [
            operation
            for operation in circuit.all_operations()
            if not isinstance(operation.gate, cirq.MeasurementGate)
        ]

        for round_index, (operation, scheduled) in enumerate(
            zip(logical_operations, plan.scheduled_rounds)
        ):
            for logical_index, path in scheduled.migrations:
                for source_module, destination_module in zip(path, path[1:]):
                    state = self._teleport(
                        logical_index,
                        source_module,
                        destination_module,
                        module_blocks,
                        current_blocks,
                        current_modules,
                        all_qubits,
                        state,
                    )

            logical_indices = plan.verified_plan.rounds[round_index].logical_qubit_indices
            gate_circuit = cirq.Circuit()
            if operation.gate == cirq.X:
                gate_circuit.append(transversal_x(current_blocks[logical_indices[0]]))
            elif operation.gate == cirq.H:
                gate_circuit.append(transversal_h(current_blocks[logical_indices[0]]))
            elif operation.gate == cirq.CNOT:
                gate_circuit.append(
                    transversal_cx(
                        current_blocks[logical_indices[0]],
                        current_blocks[logical_indices[1]],
                    )
                )
            result = self._simulate_stage(gate_circuit, all_qubits, state)
            state = result._final_simulator_state

            for logical_index in logical_indices:
                module_id = current_modules[logical_index]
                state = self._teleport(
                    logical_index,
                    module_id,
                    module_id,
                    module_blocks,
                    current_blocks,
                    current_modules,
                    all_qubits,
                    state,
                )

        measurement = next(
            operation
            for operation in circuit.all_operations()
            if isinstance(operation.gate, cirq.MeasurementGate)
        )
        measured_indices = [
            plan.verified_plan.logical_qubits.index(qubit) for qubit in measurement.qubits
        ]
        measurement_circuit = cirq.Circuit(
            cirq.measure(
                *(qubit for index in measured_indices for qubit in current_blocks[index]),
                key=cirq.measurement_key_name(measurement),
            )
        )
        result = self._simulate_stage(measurement_circuit, all_qubits, state)
        physical = result.measurements[cirq.measurement_key_name(measurement)]
        return np.asarray(
            [
                physical[index * STEANE_CODE_SIZE : (index + 1) * STEANE_CODE_SIZE]
                for index in range(len(measured_indices))
            ]
        )

    def run(
        self,
        circuit: cirq.AbstractCircuit,
        *,
        repetitions: int = 1,
    ) -> cirq.ResultDict:
        """Executes routed verified operations on fixed reusable module pools."""
        self.last_plan = compile_local_verified(
            circuit,
            self.architecture,
            self.placement,
            preparation_attempts=self.preparation_attempts,
        )
        self.decoder_events.clear()
        self.classical_messages.clear()
        self.executed_reset_count = 0
        measurement_key = self.last_plan.verified_plan.measurement_key
        measured_count = len(self.last_plan.verified_plan.measured_logical_qubit_indices)
        decoded = np.empty((repetitions, measured_count), dtype=np.int8)
        for repetition in range(repetitions):
            physical_blocks = self._run_once(circuit)
            for index, physical in enumerate(physical_blocks):
                decoded[repetition, index] = physical_measurements_to_logical_measurements(physical)
        return cirq.ResultDict(
            params=cirq.ParamResolver({}),
            measurements={measurement_key: decoded},
        )


class LocalLogicalCliffordTSimulator:
    """General logical backend for accepted local Clifford+T execution plans.

    The encoded factory and gadget hierarchy is validated by
    :func:`compile_local_verified`; simulation then occurs at the logical level,
    avoiding an intractable state vector over the encoded distillation factory.
    """

    def __init__(
        self,
        architecture: LocalFTArchitecture,
        placement: Mapping[cirq.Qid, str],
        *,
        magic_state_source: EncodedMagicStateSource | None = None,
        preparation_attempts: int = 1,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ) -> None:
        self.architecture = architecture
        self.placement = dict(placement)
        self.magic_state_source = (
            DistilledSteaneMagicStateSource() if magic_state_source is None else magic_state_source
        )
        self.preparation_attempts = preparation_attempts
        self._simulator = cirq.Simulator(seed=seed)
        self.last_plan: LocalVerifiedExecutionPlan | None = None

    def run(
        self,
        circuit: cirq.AbstractCircuit,
        *,
        repetitions: int = 1,
    ) -> cirq.Result:
        """Validates the encoded local plan and simulates its accepted logical action."""
        self.last_plan = compile_local_verified(
            circuit,
            self.architecture,
            self.placement,
            preparation_attempts=self.preparation_attempts,
            magic_state_source=self.magic_state_source,
        )
        return self._simulator.run(circuit, repetitions=repetitions)


def compile(circuit: cirq.AbstractCircuit) -> cirq.Circuit:
    """Compiles a logical circuit using Knill-style error-correcting teleportation."""
    logical_qubits = sorted(circuit.all_qubits())
    block_count = len(logical_qubits)
    physical_qubits = cirq.LineQubit.range(3 * STEANE_CODE_SIZE * block_count)

    def make_blocks(offset: int) -> dict[cirq.Qid, list[cirq.LineQubit]]:
        return {
            qubit: physical_qubits[
                offset + index * STEANE_CODE_SIZE : offset + (index + 1) * STEANE_CODE_SIZE
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
                *(physical for qubit in measured_qubits for physical in current_blocks[qubit]),
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
                    decoded[repetition, logical_index] = (
                        physical_measurements_to_logical_measurements(
                            row[start : start + STEANE_CODE_SIZE]
                        )
                    )
            logical_measurements[key] = decoded

        return cirq.ResultDict(params=physical_result.params, measurements=logical_measurements)


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
