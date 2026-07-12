import cirq
import numpy as np
import pytest
import qualtran

from cirq_superstaq import ft_compiler


def _is_within_one_bit(
    measurements: tuple[int, ...], codewords: tuple[tuple[int, ...], ...]
) -> bool:
    return any(
        sum(measured != expected for measured, expected in zip(measurements, codeword)) <= 1
        for codeword in codewords
    )


def _assert_accepted_steane_zero_passes_one_filter(
    ideal_outcomes: tuple[tuple[int, ...], ...],
    final_fault: cirq.PauliString,
    candidate: tuple[cirq.Qid, ...],
    verifier: tuple[cirq.Qid, ...],
    *,
    x_basis: bool,
    fault_description: str,
) -> None:
    if x_basis:
        final_fault = final_fault.after(cirq.H.on_each(*candidate))
    flipped_qubits = {
        qubit for qubit, pauli in final_fault.items() if pauli in (cirq.X, cirq.Y)
    }
    allowed_candidate_codewords = (
        ft_compiler._LOGICAL_ZERO_CODEWORDS + ft_compiler._LOGICAL_ONE_CODEWORDS
        if x_basis
        else ft_compiler._LOGICAL_ZERO_CODEWORDS
    )

    for ideal_outcome in ideal_outcomes:
        measured_outcome = tuple(
            bit ^ (qubit in flipped_qubits)
            for bit, qubit in zip(ideal_outcome, candidate + verifier)
        )
        candidate_measurement = measured_outcome[:7]
        verifier_measurement = measured_outcome[7:]
        if ft_compiler.steane_zero_verification_passed(verifier_measurement):
            assert _is_within_one_bit(
                candidate_measurement, allowed_candidate_codewords
            ), (
                f"Accepted output failed the one-error filter after {fault_description} "
                f"in the {'X' if x_basis else 'Z'} basis: {candidate_measurement}"
            )


def _single_pauli_faults(qubits: tuple[cirq.Qid, ...]):
    paulis = (cirq.I, cirq.X, cirq.Y, cirq.Z)
    for pauli_indices in np.ndindex(*(4,) * len(qubits)):
        if all(index == 0 for index in pauli_indices):
            continue
        operations = [paulis[index](qubit) for index, qubit in zip(pauli_indices, qubits)]
        yield operations, cirq.PauliString(
            {
                operation.qubits[0]: operation.gate
                for operation in operations
                if operation.gate != cirq.I
            }
        )


def _make_knill_teleportation_test_circuits():
    data, bell_a, bell_b = (
        tuple(cirq.LineQubit.range(start, start + 7)) for start in (0, 7, 14)
    )
    bloq = ft_compiler.KnillTeleportation(measurement_key_prefix="audit")
    operation = bloq.on_registers(
        data=np.asarray(data),
        bell_a=np.asarray(bell_a),
        bell_b=np.asarray(bell_b),
    )
    teleportation = cirq.Circuit(cirq.decompose_once(bloq, qubits=operation.qubits))
    preparation = cirq.Circuit(
        cirq.Circuit.zip(
            ft_compiler.encode(data),
            ft_compiler.encode(bell_a),
            ft_compiler.encode(bell_b),
        ),
        ft_compiler.transversal_h(bell_a),
        ft_compiler.transversal_cx(bell_a, bell_b),
    )
    return data, bell_a, bell_b, preparation, teleportation


def _assert_knill_output_passes_one_filter(
    circuit: cirq.Circuit,
    bell_b: tuple[cirq.Qid, ...],
    *,
    x_basis: bool,
    expected_logical_value: int | None,
    description: str,
) -> None:
    measured_circuit = circuit.copy()
    if x_basis:
        measured_circuit.append(ft_compiler.transversal_h(bell_b))
    measured_circuit.append(cirq.measure(*bell_b, key="audited_output"))
    result = cirq.CliffordSimulator(seed=1234).run(measured_circuit, repetitions=3)
    if expected_logical_value is None:
        expected_codewords = (
            ft_compiler._LOGICAL_ZERO_CODEWORDS + ft_compiler._LOGICAL_ONE_CODEWORDS
        )
    elif expected_logical_value:
        expected_codewords = ft_compiler._LOGICAL_ONE_CODEWORDS
    else:
        expected_codewords = ft_compiler._LOGICAL_ZERO_CODEWORDS

    for row in result.measurements["audited_output"]:
        assert _is_within_one_bit(tuple(row), expected_codewords), (
            f"Knill output failed the one-error filter after {description} "
            f"in the {'X' if x_basis else 'Z'} basis: {tuple(row)}"
        )


def test_verified_steane_zero_bloq() -> None:
    candidate, verifier = (
        np.asarray(cirq.LineQubit.range(start, start + 7)) for start in (0, 7)
    )
    bloq = ft_compiler.VerifiedSteaneZero(verification_key="verify_zero")

    assert bloq.signature == qualtran.Signature.build(candidate=7, verifier=7)
    operation = bloq.on_registers(candidate=candidate, verifier=verifier)
    lowered_operations = cirq.decompose_once(bloq, qubits=operation.qubits)
    circuit = cirq.Circuit(
        lowered_operations,
        cirq.measure(*candidate, key="result"),
    )

    result = cirq.CliffordSimulator().run(circuit, repetitions=10)

    assert all(
        ft_compiler.steane_zero_verification_passed(row)
        for row in result.measurements["verify_zero"]
    )
    assert all(
        ft_compiler.physical_measurements_to_logical_measurements(row) == 0
        for row in result.measurements["result"]
    )
    assert cirq.measurement_key_names(circuit) == {"verify_zero", "result"}


def test_verified_steane_plus_composes_verified_zero() -> None:
    candidate, verifier = (
        np.asarray(cirq.LineQubit.range(start, start + 7)) for start in (0, 7)
    )
    bloq = ft_compiler.VerifiedSteanePlus(verification_key="verify_plus")

    assert bloq.signature == qualtran.Signature.build(candidate=7, verifier=7)
    operation = bloq.on_registers(candidate=candidate, verifier=verifier)
    outer_operations = list(
        cirq.flatten_op_tree(cirq.decompose_once(bloq, qubits=operation.qubits))
    )

    zero_preparation = outer_operations[0]
    assert isinstance(zero_preparation.gate, ft_compiler.VerifiedSteaneZero)
    assert zero_preparation.gate.verification_key == "verify_plus"

    lowered_zero = cirq.decompose_once(
        zero_preparation.gate,
        qubits=zero_preparation.qubits,
    )
    circuit = cirq.Circuit(
        lowered_zero,
        outer_operations[1:],
        ft_compiler.transversal_h(candidate),
        cirq.measure(*candidate, key="result"),
    )
    result = cirq.CliffordSimulator().run(circuit, repetitions=10)

    assert all(
        ft_compiler.steane_zero_verification_passed(row)
        for row in result.measurements["verify_plus"]
    )
    assert all(
        ft_compiler.physical_measurements_to_logical_measurements(row) == 0
        for row in result.measurements["result"]
    )


def test_verified_steane_bell_pair_composes_verified_states() -> None:
    bell_a, bell_a_verifier, bell_b, bell_b_verifier = (
        np.asarray(cirq.LineQubit.range(start, start + 7))
        for start in (0, 7, 14, 21)
    )
    bloq = ft_compiler.VerifiedSteaneBellPair(
        plus_verification_key="verify_bell_plus",
        zero_verification_key="verify_bell_zero",
    )

    assert bloq.signature == qualtran.Signature.build(
        bell_a=7,
        bell_a_verifier=7,
        bell_b=7,
        bell_b_verifier=7,
    )
    operation = bloq.on_registers(
        bell_a=bell_a,
        bell_a_verifier=bell_a_verifier,
        bell_b=bell_b,
        bell_b_verifier=bell_b_verifier,
    )
    outer_operations = list(
        cirq.flatten_op_tree(cirq.decompose_once(bloq, qubits=operation.qubits))
    )

    plus_preparation, zero_preparation = outer_operations[:2]
    assert isinstance(plus_preparation.gate, ft_compiler.VerifiedSteanePlus)
    assert isinstance(zero_preparation.gate, ft_compiler.VerifiedSteaneZero)

    plus_operations = list(
        cirq.flatten_op_tree(
            cirq.decompose_once(plus_preparation.gate, qubits=plus_preparation.qubits)
        )
    )
    plus_zero_preparation = plus_operations[0]
    assert isinstance(plus_zero_preparation.gate, ft_compiler.VerifiedSteaneZero)

    circuit = cirq.Circuit(
        cirq.decompose_once(
            plus_zero_preparation.gate,
            qubits=plus_zero_preparation.qubits,
        ),
        plus_operations[1:],
        cirq.decompose_once(
            zero_preparation.gate,
            qubits=zero_preparation.qubits,
        ),
        outer_operations[2:],
        cirq.measure(*bell_a, key="bell_a"),
        cirq.measure(*bell_b, key="bell_b"),
    )
    result = cirq.CliffordSimulator().run(circuit, repetitions=20)

    assert all(
        ft_compiler.steane_bell_pair_verification_passed(plus_row, zero_row)
        for plus_row, zero_row in zip(
            result.measurements["verify_bell_plus"],
            result.measurements["verify_bell_zero"],
        )
    )
    logical_a = np.asarray(
        [
            ft_compiler.physical_measurements_to_logical_measurements(row)
            for row in result.measurements["bell_a"]
        ]
    )
    logical_b = np.asarray(
        [
            ft_compiler.physical_measurements_to_logical_measurements(row)
            for row in result.measurements["bell_b"]
        ]
    )
    np.testing.assert_array_equal(logical_a, logical_b)


@pytest.mark.parametrize("logical_value", (0, 1))
def test_knill_teleportation_consumes_accepted_bell_pair(logical_value: int) -> None:
    data, bell_a, bell_b = (
        np.asarray(cirq.LineQubit.range(start, start + 7)) for start in (0, 7, 14)
    )
    bloq = ft_compiler.KnillTeleportation(measurement_key_prefix="teleport")

    assert bloq.signature == qualtran.Signature.build(data=7, bell_a=7, bell_b=7)
    operation = bloq.on_registers(data=data, bell_a=bell_a, bell_b=bell_b)
    lowered_operations = cirq.decompose_once(bloq, qubits=operation.qubits)
    circuit = cirq.Circuit(
        cirq.Circuit.zip(
            ft_compiler.encode(data),
            ft_compiler.encode(bell_a),
            ft_compiler.encode(bell_b),
        ),
        ft_compiler.transversal_x(data) if logical_value else [],
        ft_compiler.transversal_h(bell_a),
        ft_compiler.transversal_cx(bell_a, bell_b),
        lowered_operations,
        cirq.measure(*bell_b, key="result"),
    )

    result = ft_compiler.qec_simulator(circuit, repetitions=10)

    np.testing.assert_array_equal(
        result.measurements["result"],
        np.full((10, 1), logical_value, dtype=np.int8),
    )
    assert cirq.measurement_key_names(circuit) == {
        "teleport_data",
        "teleport_bell_a",
        "result",
    }


def test_steane_bell_measurement_decoder_returns_physical_and_logical_updates() -> None:
    logical_one = np.asarray(ft_compiler._LOGICAL_ONE_CODEWORDS[0], dtype=np.int8)
    for error_index in range(7):
        data_measurements = logical_one.copy()
        data_measurements[error_index] ^= 1
        bell_a_measurements = np.zeros(7, dtype=np.int8)
        bell_a_measurements[error_index] = 1

        update = ft_compiler.SteaneBellMeasurementDecoder.decode(
            data_measurements,
            bell_a_measurements,
        )

        assert update == ft_compiler.SteanePauliFrameUpdate(
            physical_x_index=error_index,
            physical_z_index=error_index,
            logical_x=False,
            logical_z=True,
        )


@pytest.mark.parametrize(
    "error_index, pauli",
    [(index, pauli) for index in range(7) for pauli in (cirq.X, cirq.Y, cirq.Z)],
)
def test_knill_teleportation_corrects_single_input_error(
    error_index: int, pauli: cirq.Pauli
) -> None:
    data, bell_a, bell_b = (
        np.asarray(cirq.LineQubit.range(start, start + 7)) for start in (0, 7, 14)
    )
    bloq = ft_compiler.KnillTeleportation(measurement_key_prefix="correct")
    operation = bloq.on_registers(data=data, bell_a=bell_a, bell_b=bell_b)
    lowered_operations = cirq.decompose_once(bloq, qubits=operation.qubits)
    preparation = cirq.Circuit(
        cirq.Circuit.zip(
            ft_compiler.encode(data),
            ft_compiler.encode(bell_a),
            ft_compiler.encode(bell_b),
        ),
        pauli(data[error_index]),
        ft_compiler.transversal_h(bell_a),
        ft_compiler.transversal_cx(bell_a, bell_b),
        lowered_operations,
    )

    for x_basis in (False, True):
        circuit = preparation.copy()
        if x_basis:
            circuit.append(ft_compiler.transversal_h(bell_b))
        circuit.append(cirq.measure(*bell_b, key="result"))
        result = cirq.CliffordSimulator().run(circuit, repetitions=5)
        allowed_codewords = (
            ft_compiler._LOGICAL_ZERO_CODEWORDS + ft_compiler._LOGICAL_ONE_CODEWORDS
            if x_basis
            else ft_compiler._LOGICAL_ZERO_CODEWORDS
        )

        assert all(tuple(row) in allowed_codewords for row in result.measurements["result"])


@pytest.mark.parametrize(
    "logical_state, input_operations, x_basis, expected_value",
    (
        ("zero", (), False, 0),
        ("one", ("X",), False, 1),
        ("plus", ("H",), True, 0),
        ("minus", ("X", "H"), True, 1),
    ),
)
def test_knill_teleportation_preserves_logical_states(
    logical_state: str,
    input_operations: tuple[str, ...],
    x_basis: bool,
    expected_value: int,
) -> None:
    data, _, bell_b, preparation, teleportation = (
        _make_knill_teleportation_test_circuits()
    )
    input_circuit = cirq.Circuit()
    for gate_name in input_operations:
        gate = cirq.X if gate_name == "X" else cirq.H
        input_circuit.append(gate.on_each(*data))

    _assert_knill_output_passes_one_filter(
        preparation + input_circuit + teleportation,
        bell_b,
        x_basis=x_basis,
        expected_logical_value=expected_value,
        description=f"ideal teleportation of logical {logical_state}",
    )


def test_knill_teleportation_audits_accepted_bell_resource_errors() -> None:
    _, bell_a, bell_b, preparation, teleportation = (
        _make_knill_teleportation_test_circuits()
    )
    resource_faults: list[tuple[str, list[cirq.Operation]]] = []

    for block_name, block in (("bell_a", bell_a), ("bell_b", bell_b)):
        for qubit in block:
            for pauli in (cirq.X, cirq.Y, cirq.Z):
                resource_faults.append(
                    (f"{pauli} error on accepted {block_name} qubit {qubit}", [pauli(qubit)])
                )

    for bell_a_qubit, bell_b_qubit in zip(bell_a, bell_b):
        for fault_operations, _ in _single_pauli_faults((bell_a_qubit, bell_b_qubit)):
            resource_faults.append(
                (
                    f"correlated accepted-resource error {fault_operations!r}",
                    fault_operations,
                )
            )

    for description, fault_operations in resource_faults:
        circuit = preparation + cirq.Circuit(fault_operations) + teleportation
        for x_basis in (False, True):
            _assert_knill_output_passes_one_filter(
                circuit,
                bell_b,
                x_basis=x_basis,
                expected_logical_value=None if x_basis else 0,
                description=description,
            )


def test_knill_teleportation_audits_internal_faults() -> None:
    _, _, bell_b, preparation, teleportation = (
        _make_knill_teleportation_test_circuits()
    )
    internal_faults: list[tuple[str, cirq.Circuit]] = []

    # Exhaust all Pauli faults following the unitary locations in the
    # transversal Bell measurement.
    for moment_index, operation in teleportation.findall_operations(
        lambda operation: not cirq.is_measurement(operation)
        and not operation.classical_controls
    ):
        if operation.gate not in (cirq.CNOT, cirq.H):
            continue
        for fault_operations, _ in _single_pauli_faults(operation.qubits):
            faulted = teleportation.copy()
            faulted.insert(
                moment_index + 1,
                fault_operations,
                strategy=cirq.InsertStrategy.NEW,
            )
            internal_faults.append(
                (f"fault {fault_operations!r} after {operation!r}", faulted)
            )

    # A physical measurement fault is equivalent to flipping that measurement
    # result immediately before an otherwise ideal Z-basis measurement.
    for moment_index, measurement in teleportation.findall_operations(
        cirq.is_measurement
    ):
        for qubit in measurement.qubits:
            faulted = teleportation.copy()
            faulted.insert(
                moment_index,
                cirq.X(qubit),
                strategy=cirq.InsertStrategy.NEW,
            )
            internal_faults.append((f"measurement fault on {qubit}", faulted))

    for description, faulted_teleportation in internal_faults:
        circuit = preparation + faulted_teleportation
        for x_basis in (False, True):
            _assert_knill_output_passes_one_filter(
                circuit,
                bell_b,
                x_basis=x_basis,
                expected_logical_value=None if x_basis else 0,
                description=description,
            )

    # A single fault in any physical Pauli correction acts on only one output
    # qubit. Check the complete single-qubit Pauli set independently of whether
    # that classically controlled correction fires in a particular shot.
    ideal_circuit = preparation + teleportation
    for qubit in bell_b:
        for pauli in (cirq.X, cirq.Y, cirq.Z):
            circuit = ideal_circuit + cirq.Circuit(pauli(qubit))
            for x_basis in (False, True):
                _assert_knill_output_passes_one_filter(
                    circuit,
                    bell_b,
                    x_basis=x_basis,
                    expected_logical_value=None if x_basis else 0,
                    description=f"{pauli} fault during output correction on {qubit}",
                )


def test_verified_knill_ec_separates_preparation_from_data_interaction() -> None:
    protocol = ft_compiler.VerifiedKnillEC(key_prefix="round_0")

    bell_pair = protocol.bell_pair
    assert isinstance(bell_pair, ft_compiler.VerifiedSteaneBellPair)
    assert bell_pair.plus_verification_key == "round_0_plus_verification"
    assert bell_pair.zero_verification_key == "round_0_zero_verification"

    teleportation = protocol.teleportation
    assert isinstance(teleportation, ft_compiler.KnillTeleportation)
    assert teleportation.measurement_key_prefix == "round_0_teleportation"

    accepted_zero = np.zeros(7, dtype=np.int8)
    assert protocol.verification_passed(
        {
            protocol.plus_verification_key: accepted_zero,
            protocol.zero_verification_key: accepted_zero,
        }
    )

    rejected_zero = accepted_zero.copy()
    rejected_zero[3] = 1
    assert not protocol.verification_passed(
        {
            protocol.plus_verification_key: accepted_zero,
            protocol.zero_verification_key: rejected_zero,
        }
    )


def test_verified_steane_bell_pair_factory_selects_first_accepted_attempt() -> None:
    factory = ft_compiler.VerifiedSteaneBellPairFactory(
        attempts=3,
        key_prefix="factory",
    )

    preparations = factory.preparations
    assert len(preparations) == 3
    assert [preparation.plus_verification_key for preparation in preparations] == [
        "factory_0_plus_verification",
        "factory_1_plus_verification",
        "factory_2_plus_verification",
    ]

    accepted = np.zeros(7, dtype=np.int8)
    rejected = accepted.copy()
    rejected[0] = 1
    measurements = {
        "factory_0_plus_verification": accepted,
        "factory_0_zero_verification": rejected,
        "factory_1_plus_verification": accepted,
        "factory_1_zero_verification": accepted,
        "factory_2_plus_verification": accepted,
        "factory_2_zero_verification": accepted,
    }

    assert factory.select(measurements) == 1
    assert factory.protocol(1).teleportation.measurement_key_prefix == (
        "factory_1_teleportation"
    )


def test_verified_steane_bell_pair_factory_rejects_failed_batch() -> None:
    factory = ft_compiler.VerifiedSteaneBellPairFactory(attempts=2)
    rejected = np.asarray([1, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    measurements = {
        key: rejected
        for preparation in factory.preparations
        for key in (
            preparation.plus_verification_key,
            preparation.zero_verification_key,
        )
    }

    with pytest.raises(RuntimeError, match="No verified Steane Bell-pair"):
        factory.select(measurements)


def test_verified_steane_bell_pair_factory_validates_attempt_count() -> None:
    with pytest.raises(ValueError, match="at least one attempt"):
        ft_compiler.VerifiedSteaneBellPairFactory(attempts=0)


def test_compile_verified_builds_staged_execution_plan() -> None:
    logical_qubit = cirq.LineQubit(0)
    logical_circuit = cirq.Circuit(
        cirq.X(logical_qubit),
        cirq.measure(logical_qubit, key="result"),
    )

    plan = ft_compiler.compile_verified(logical_circuit, preparation_attempts=2)

    preparation_operations = tuple(plan.resource_preparation_circuit.all_operations())
    assert len(preparation_operations) == 2
    assert all(
        isinstance(operation.gate, ft_compiler.VerifiedSteaneBellPair)
        for operation in preparation_operations
    )
    assert set(plan.data).isdisjoint(plan.resource_preparation_circuit.all_qubits())

    accepted = np.zeros(7, dtype=np.int8)
    rejected = accepted.copy()
    rejected[2] = 1
    continuation = plan.build_continuation(
        {
            "verified_compile_round_0_logical_0_0_plus_verification": accepted,
            "verified_compile_round_0_logical_0_0_zero_verification": rejected,
            "verified_compile_round_0_logical_0_1_plus_verification": accepted,
            "verified_compile_round_0_logical_0_1_zero_verification": accepted,
        }
    )

    teleportation = next(
        operation
        for operation in continuation.all_operations()
        if isinstance(operation.gate, ft_compiler.KnillTeleportation)
    )
    selected_blocks = plan.rounds[0].resources[0].bell_pair_blocks[1]
    assert teleportation.qubits == plan.data + selected_blocks.bell_a + selected_blocks.bell_b
    output_measurement = next(
        operation
        for operation in continuation.all_operations()
        if isinstance(operation.gate, cirq.MeasurementGate)
    )
    assert output_measurement.qubits == selected_blocks.bell_b
    assert cirq.measurement_key_name(output_measurement) == "result"


def test_compile_verified_executes_selected_ideal_resource() -> None:
    logical_qubit = cirq.LineQubit(0)
    plan = ft_compiler.compile_verified(
        cirq.Circuit(cirq.X(logical_qubit), cirq.measure(logical_qubit, key="result")),
        preparation_attempts=1,
    )
    preparation_operation = next(plan.resource_preparation_circuit.all_operations())
    preparation_gate = preparation_operation.gate
    assert isinstance(preparation_gate, ft_compiler.VerifiedSteaneBellPair)
    outer_preparation = list(
        cirq.flatten_op_tree(
            cirq.decompose_once(preparation_gate, qubits=preparation_operation.qubits)
        )
    )
    plus_preparation, zero_preparation = outer_preparation[:2]
    plus_operations = list(
        cirq.flatten_op_tree(
            cirq.decompose_once(plus_preparation.gate, qubits=plus_preparation.qubits)
        )
    )
    plus_zero = plus_operations[0]
    lowered_preparation = cirq.Circuit(
        cirq.decompose_once(plus_zero.gate, qubits=plus_zero.qubits),
        plus_operations[1:],
        cirq.decompose_once(zero_preparation.gate, qubits=zero_preparation.qubits),
        outer_preparation[2:],
    )

    accepted = np.zeros(7, dtype=np.int8)
    continuation = plan.build_continuation(
        {
            "verified_compile_round_0_logical_0_0_plus_verification": accepted,
            "verified_compile_round_0_logical_0_0_zero_verification": accepted,
        }
    )
    lowered_continuation = cirq.Circuit()
    for operation in continuation.all_operations():
        if isinstance(operation.gate, ft_compiler.KnillTeleportation):
            lowered_continuation.append(
                cirq.decompose_once(operation.gate, qubits=operation.qubits)
            )
        else:
            lowered_continuation.append(operation)

    result = ft_compiler.qec_simulator(
        lowered_preparation + lowered_continuation,
        repetitions=5,
    )

    np.testing.assert_array_equal(result.measurements["result"], np.ones((5, 1)))


def test_compile_verified_chains_arbitrary_depth_rounds() -> None:
    logical_qubit = cirq.LineQubit(0)
    plan = ft_compiler.compile_verified(
        cirq.Circuit(
            cirq.H(logical_qubit),
            cirq.X(logical_qubit),
            cirq.H(logical_qubit),
            cirq.measure(logical_qubit, key="result"),
        ),
        preparation_attempts=1,
    )

    assert [round_.logical_gate for round_ in plan.rounds] == [cirq.H, cirq.X, cirq.H]
    assert len(tuple(plan.resource_preparation_circuit.all_operations())) == 3

    accepted = np.zeros(7, dtype=np.int8)
    continuation = plan.build_continuation(
        {
            f"verified_compile_round_{round_index}_logical_0_0_{state}_verification": accepted
            for round_index in range(3)
            for state in ("plus", "zero")
        }
    )
    teleportations = [
        operation
        for operation in continuation.all_operations()
        if isinstance(operation.gate, ft_compiler.KnillTeleportation)
    ]

    assert len(teleportations) == 3
    assert teleportations[0].qubits[:7] == plan.data
    assert teleportations[1].qubits[:7] == (
        plan.rounds[0].resources[0].bell_pair_blocks[0].bell_b
    )
    assert teleportations[2].qubits[:7] == (
        plan.rounds[1].resources[0].bell_pair_blocks[0].bell_b
    )
    output_measurement = next(
        operation
        for operation in continuation.all_operations()
        if isinstance(operation.gate, cirq.MeasurementGate)
    )
    assert output_measurement.qubits == plan.rounds[2].resources[0].bell_pair_blocks[0].bell_b


def test_compile_verified_supports_multiple_logical_blocks_and_cnot() -> None:
    control, target = cirq.LineQubit.range(2)
    plan = ft_compiler.compile_verified(
        cirq.Circuit(
            cirq.X(control),
            cirq.CNOT(control, target),
            cirq.measure(control, target, key="result"),
        ),
        preparation_attempts=1,
    )

    assert len(plan.data_blocks) == 2
    assert plan.rounds[0].logical_qubit_indices == (0,)
    assert plan.rounds[1].logical_qubit_indices == (0, 1)
    assert len(tuple(plan.resource_preparation_circuit.all_operations())) == 3

    accepted = np.zeros(7, dtype=np.int8)
    continuation = plan.build_continuation(
        {
            f"verified_compile_round_{round_index}_logical_{logical_index}_0_"
            f"{state}_verification": accepted
            for round_index, logical_indices in ((0, (0,)), (1, (0, 1)))
            for logical_index in logical_indices
            for state in ("plus", "zero")
        }
    )
    teleportations = [
        operation
        for operation in continuation.all_operations()
        if isinstance(operation.gate, ft_compiler.KnillTeleportation)
    ]
    assert len(teleportations) == 3

    control_before_cnot = plan.rounds[0].resources[0].bell_pair_blocks[0].bell_b
    target_before_cnot = plan.data_blocks[1]
    logical_cnot_operations = [
        operation
        for operation in continuation.all_operations()
        if operation.gate == cirq.CNOT
        and operation.qubits[0] in control_before_cnot
        and operation.qubits[1] in target_before_cnot
    ]
    assert len(logical_cnot_operations) == 7

    control_output = plan.rounds[1].resources[0].bell_pair_blocks[0].bell_b
    target_output = plan.rounds[1].resources[1].bell_pair_blocks[0].bell_b
    output_measurement = next(
        operation
        for operation in continuation.all_operations()
        if isinstance(operation.gate, cirq.MeasurementGate)
    )
    assert output_measurement.qubits == control_output + target_output


def test_verified_ft_simulator_executes_staged_single_qubit_plan() -> None:
    qubit = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(qubit), cirq.measure(qubit, key="result"))

    result = ft_compiler.VerifiedFTSimulator(
        preparation_attempts=1,
        seed=1234,
    ).run(circuit, repetitions=3)

    np.testing.assert_array_equal(result.measurements["result"], np.ones((3, 1)))


def test_verified_ft_simulator_executes_multiblock_cnot() -> None:
    control, target = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.X(control),
        cirq.CNOT(control, target),
        cirq.measure(control, target, key="result"),
    )

    result = ft_compiler.VerifiedFTSimulator(
        preparation_attempts=1,
        seed=1234,
    ).run(circuit, repetitions=3)

    np.testing.assert_array_equal(result.measurements["result"], np.ones((3, 2)))


def test_compile_local_verified_enforces_module_links_and_capacity() -> None:
    control, target = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(control),
        cirq.CNOT(control, target),
        cirq.measure(control, target, key="result"),
    )
    architecture = ft_compiler.LocalFTArchitecture(
        modules=(
            ft_compiler.LocalFTModule("left", bell_pair_attempt_capacity=2),
            ft_compiler.LocalFTModule("right", bell_pair_attempt_capacity=2),
        ),
        links=frozenset({("left", "right")}),
    )

    local_plan = ft_compiler.compile_local_verified(
        circuit,
        architecture,
        {control: "left", target: "right"},
        preparation_attempts=2,
    )

    assert local_plan.module_for_logical_index(0).module_id == "left"
    assert local_plan.module_for_logical_index(1).module_id == "right"

    disconnected = ft_compiler.LocalFTArchitecture(
        modules=architecture.modules,
        links=frozenset(),
    )
    with pytest.raises(ValueError, match="No local route"):
        ft_compiler.compile_local_verified(
            circuit,
            disconnected,
            {control: "left", target: "right"},
        )

    with pytest.raises(ValueError, match="attempt capacity"):
        ft_compiler.compile_local_verified(
            circuit,
            architecture,
            {control: "left", target: "right"},
            preparation_attempts=3,
        )


def test_compile_local_verified_routes_and_reuses_bounded_slots() -> None:
    control, target = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.X(control),
        cirq.CNOT(control, target),
        cirq.measure(control, target, key="result"),
    )
    architecture = ft_compiler.LocalFTArchitecture(
        modules=tuple(ft_compiler.LocalFTModule(name) for name in ("left", "middle", "right")),
        links=frozenset({("left", "middle"), ("middle", "right")}),
    )

    local_plan = ft_compiler.compile_local_verified(
        circuit,
        architecture,
        {control: "left", target: "right"},
    )

    cnot_round = local_plan.scheduled_rounds[1]
    assert cnot_round.migrations == ((0, ("left", "middle")),)
    assert local_plan.logical_module_ids == ("middle", "right")
    assert {assignment.module_id for assignment in cnot_round.resources} == {
        "middle",
        "right",
    }
    # The same physical slot is reused in later moments instead of allocating
    # resources proportional to circuit depth.
    assert local_plan.scheduled_rounds[0].resources[0].slot_index == 0
    assert cnot_round.resources[0].slot_index == 0


def test_compile_local_verified_preserves_parallel_moments() -> None:
    first, second = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.Moment(cirq.H(first), cirq.X(second)),
        cirq.measure(first, second, key="result"),
    )
    architecture = ft_compiler.LocalFTArchitecture(
        modules=(
            ft_compiler.LocalFTModule(
                "module",
                logical_capacity=2,
                bell_pair_slots=2,
            ),
        ),
        links=frozenset(),
    )

    local_plan = ft_compiler.compile_local_verified(
        circuit,
        architecture,
        {first: "module", second: "module"},
    )

    assert [round_.moment_index for round_ in local_plan.scheduled_rounds] == [0, 0]
    assert [round_.resources[0].slot_index for round_ in local_plan.scheduled_rounds] == [0, 1]


def test_local_architecture_qubit_bound_is_independent_of_depth() -> None:
    qubit = cirq.LineQubit(0)
    architecture = ft_compiler.LocalFTArchitecture(
        modules=(ft_compiler.LocalFTModule("module"),),
        links=frozenset(),
    )
    shallow = ft_compiler.compile_local_verified(
        cirq.Circuit(cirq.X(qubit), cirq.measure(qubit, key="result")),
        architecture,
        {qubit: "module"},
    )
    deep = ft_compiler.compile_local_verified(
        cirq.Circuit(
            *(cirq.H(qubit) if index % 2 else cirq.X(qubit) for index in range(20)),
            cirq.measure(qubit, key="result"),
        ),
        architecture,
        {qubit: "module"},
    )

    assert shallow.physical_qubit_bound == deep.physical_qubit_bound == 35


def test_local_verified_ft_simulator_runs_two_module_bell_demo() -> None:
    control, target = cirq.LineQubit.range(2)
    architecture = ft_compiler.LocalFTArchitecture(
        modules=(ft_compiler.LocalFTModule("left"), ft_compiler.LocalFTModule("right")),
        links=frozenset({("left", "right")}),
    )
    simulator = ft_compiler.LocalVerifiedFTSimulator(
        architecture,
        {control: "left", target: "right"},
        seed=1234,
    )
    result = simulator.run(
        cirq.Circuit(
            cirq.H(control),
            cirq.CNOT(control, target),
            cirq.measure(control, target, key="result"),
        ),
        repetitions=10,
    )

    np.testing.assert_array_equal(
        result.measurements["result"][:, 0],
        result.measurements["result"][:, 1],
    )
    assert simulator.last_plan is not None
    assert simulator.last_plan.physical_qubit_bound == 70
    assert simulator.physical_qubits_used == 70
    assert simulator.executed_reset_count > 0


def test_local_verified_ft_simulator_executes_three_module_route() -> None:
    control, target = cirq.LineQubit.range(2)
    architecture = ft_compiler.LocalFTArchitecture(
        modules=tuple(ft_compiler.LocalFTModule(name) for name in ("A", "B", "C")),
        links=frozenset({("A", "B"), ("B", "C")}),
    )
    simulator = ft_compiler.LocalVerifiedFTSimulator(
        architecture,
        {control: "A", target: "C"},
        seed=1234,
    )
    result = simulator.run(
        cirq.Circuit(
            cirq.X(control),
            cirq.CNOT(control, target),
            cirq.measure(control, target, key="result"),
        ),
        repetitions=3,
    )

    np.testing.assert_array_equal(result.measurements["result"], np.ones((3, 2)))
    assert simulator.last_plan is not None
    assert simulator.last_plan.scheduled_rounds[1].migrations == ((0, ("A", "B")),)
    assert simulator.physical_qubits_used == simulator.last_plan.physical_qubit_bound == 105
    assert any(
        message.source_module == "A" and message.destination_module == "B"
        for message in simulator.classical_messages
    )
    assert all(
        architecture.are_neighbors(message.source_module, message.destination_module)
        for message in simulator.classical_messages
    )
    assert {event.module_id for event in simulator.decoder_events}.issubset({"A", "B", "C"})


def test_local_verified_ft_simulator_reuses_same_pool_across_depth() -> None:
    qubit = cirq.LineQubit(0)
    architecture = ft_compiler.LocalFTArchitecture(
        modules=(ft_compiler.LocalFTModule("module"),),
        links=frozenset(),
    )
    simulator = ft_compiler.LocalVerifiedFTSimulator(
        architecture,
        {qubit: "module"},
        seed=1234,
    )
    circuit = cirq.Circuit(
        *(cirq.X(qubit) for _ in range(6)),
        cirq.measure(qubit, key="result"),
    )

    result = simulator.run(circuit, repetitions=2)

    np.testing.assert_array_equal(result.measurements["result"], np.zeros((2, 1)))
    assert simulator.physical_qubits_used == 35
    assert simulator.executed_reset_count == 2 * 6 * 4 * 7


def test_logical_cnot_extended_rectangle_single_fault_contract() -> None:
    control = tuple(cirq.LineQubit.range(0, 7))
    target = tuple(cirq.LineQubit.range(7, 14))
    transversal_cnot = ft_compiler.transversal_cx(control, target)

    def assert_correctable_by_independent_knill_ec(error: cirq.PauliString) -> None:
        error_qubits = set(error.qubits)
        assert len(error_qubits.intersection(control)) <= 1
        assert len(error_qubits.intersection(target)) <= 1

    for source_block in (control, target):
        for qubit in source_block:
            for pauli in (cirq.X, cirq.Y, cirq.Z):
                assert_correctable_by_independent_knill_ec(
                    cirq.PauliString(pauli(qubit)).after(transversal_cnot)
                )
    for control_qubit, target_qubit in zip(control, target):
        for _, fault in _single_pauli_faults((control_qubit, target_qubit)):
            assert_correctable_by_independent_knill_ec(fault)


def test_compile_verified_validates_mvp_scope() -> None:
    qubits = cirq.LineQubit.range(2)
    with pytest.raises(ValueError, match="Unsupported verified logical operation"):
        ft_compiler.compile_verified(
            cirq.Circuit(cirq.Z(qubits[0]), cirq.measure(qubits[0], key="result"))
        )

    with pytest.raises(ValueError, match="at least one logical gate"):
        ft_compiler.compile_verified(
            cirq.Circuit(cirq.measure(qubits[0], key="result"))
        )


@pytest.mark.parametrize("error_index", range(7))
def test_steane_zero_verification_rejects_single_bit_errors(error_index: int) -> None:
    erroneous_codeword = np.zeros(7, dtype=np.int8)
    erroneous_codeword[error_index] = 1

    assert not ft_compiler.steane_zero_verification_passed(erroneous_codeword)


def test_verified_steane_zero_accepted_outputs_pass_single_fault_audit() -> None:
    candidate = tuple(cirq.LineQubit.range(0, 7))
    verifier = tuple(cirq.LineQubit.range(7, 14))
    base_circuit = cirq.Circuit(
        cirq.Circuit.zip(
            ft_compiler.encode(candidate),
            ft_compiler.encode(verifier),
        ),
        ft_compiler.transversal_cx(candidate, verifier),
    )
    qubit_order = candidate + verifier

    def outcome_support(circuit: cirq.Circuit) -> tuple[tuple[int, ...], ...]:
        state_vector = cirq.final_state_vector(circuit, qubit_order=qubit_order)
        return tuple(
            tuple(cirq.big_endian_int_to_bits(outcome, bit_count=14))
            for outcome, amplitude in enumerate(state_vector)
            if abs(amplitude) ** 2 >= 1e-10
        )

    ideal_z_outcomes = outcome_support(base_circuit)
    ideal_x_outcomes = outcome_support(
        base_circuit + cirq.Circuit(cirq.H.on_each(*candidate))
    )
    fault_cases: list[tuple[str, cirq.PauliString, int]] = []

    # A faulty physical |0> preparation is represented by a Pauli immediately
    # before the encoding circuit.
    for qubit in candidate + verifier:
        for pauli in (cirq.X, cirq.Y, cirq.Z):
            fault_cases.append(
                (
                    f"{pauli} preparation fault on {qubit}",
                    cirq.PauliString(pauli(qubit)),
                    0,
                )
            )

    # A faulty one- or two-qubit Clifford location may append any non-identity
    # Pauli supported on the qubits participating in that location.
    for moment_index, operation in base_circuit.findall_operations(lambda _: True):
        for fault_operations, fault in _single_pauli_faults(operation.qubits):
            fault_cases.append(
                (
                    f"fault {fault_operations!r} after {operation!r} in moment {moment_index}",
                    fault,
                    moment_index + 1,
                )
            )

    for fault_description, fault, suffix_start in fault_cases:
        final_fault = fault.after(base_circuit[suffix_start:].all_operations())
        _assert_accepted_steane_zero_passes_one_filter(
            ideal_z_outcomes,
            final_fault,
            candidate,
            verifier,
            x_basis=False,
            fault_description=fault_description,
        )
        _assert_accepted_steane_zero_passes_one_filter(
            ideal_x_outcomes,
            final_fault,
            candidate,
            verifier,
            x_basis=True,
            fault_description=fault_description,
        )


def test_verified_steane_plus_preserves_single_fault_contract() -> None:
    block = tuple(cirq.LineQubit.range(7))
    transversal_h = ft_compiler.transversal_h(block)

    # VerifiedSteanePlus receives an accepted output from VerifiedSteaneZero.
    # Conjugation by the transversal logical H can exchange X and Z but cannot
    # increase the physical support of any correctable output error.
    for qubit in block:
        for pauli in (cirq.X, cirq.Y, cirq.Z):
            inherited_error = cirq.PauliString(pauli(qubit))
            output_error = inherited_error.after(transversal_h)

            assert len(output_error) == 1
            assert set(output_error.qubits) == {qubit}


def test_verified_steane_bell_pair_preserves_single_fault_contract() -> None:
    bell_a = tuple(cirq.LineQubit.range(0, 7))
    bell_b = tuple(cirq.LineQubit.range(7, 14))
    final_transversal_cnot = ft_compiler.transversal_cx(bell_a, bell_b)

    def assert_one_error_per_block(error: cirq.PauliString, description: str) -> None:
        error_qubits = set(error.qubits)
        assert len(error_qubits.intersection(bell_a)) <= 1, description
        assert len(error_qubits.intersection(bell_b)) <= 1, description

    # If the single fault occurred inside either accepted child preparation,
    # that child's proven contract permits one output error. A transversal CNOT
    # can copy it to the corresponding physical position in the other block,
    # but cannot create two errors in either block.
    for source_name, source_block in (("plus", bell_a), ("zero", bell_b)):
        for qubit in source_block:
            for pauli in (cirq.X, cirq.Y, cirq.Z):
                child_error = cirq.PauliString(pauli(qubit))
                output_error = child_error.after(final_transversal_cnot)
                assert_one_error_per_block(
                    output_error,
                    f"{pauli} error inherited from the verified {source_name} preparation",
                )

    # If both child preparations were fault-free, the one fault may instead be
    # at any physical CNOT in the final transversal layer. Enumerate the complete
    # non-identity two-qubit Pauli fault set for every such location.
    for control, target in zip(bell_a, bell_b):
        for _, cnot_fault in _single_pauli_faults((control, target)):
            assert_one_error_per_block(
                cnot_fault,
                f"fault at final Bell-pair CNOT on {(control, target)}",
            )


def test_compile() -> None:
    qubit = cirq.LineQubit(0)
    logical_circuit = cirq.Circuit(cirq.X(qubit), cirq.measure(qubit, key="m"))

    compiled_circuit = ft_compiler.compile(logical_circuit)

    print(compiled_circuit)

    # assert len(compiled_circuit.all_qubits()) == 21
    # assert sum(op.gate == cirq.X for op in compiled_circuit.all_operations()) == 14
    # assert sum(isinstance(op.gate, cirq.ResetChannel) for op in compiled_circuit.all_operations()) == 14
    # assert cirq.measurement_key_names(compiled_circuit) == {
    #     "__ft_data_0_0",
    #     "__ft_ancilla_0_0",
    #     "__ft_data_0_1",
    #     "__ft_ancilla_0_1",
    #     "m",
    # }
    # measurement = next(
    #     op
    #     for op in compiled_circuit.all_operations()
    #     if isinstance(op.gate, cirq.MeasurementGate) and cirq.measurement_key_name(op) == "m"
    # )
    # assert measurement.qubits == tuple(cirq.LineQubit.range(7))


def test_compile_encodes_data_and_ancillas_in_parallel() -> None:
    qubit = cirq.LineQubit(0)
    logical_circuit = cirq.Circuit(cirq.I(qubit), cirq.measure(qubit, key="m"))

    compiled_circuit = ft_compiler.compile(logical_circuit)
    expected_encoding = cirq.Circuit.zip(
        ft_compiler.encode(cirq.LineQubit.range(0, 7)),
        ft_compiler.encode(cirq.LineQubit.range(7, 14)),
        ft_compiler.encode(cirq.LineQubit.range(14, 21)),
    )

    assert compiled_circuit[: len(expected_encoding)] == expected_encoding


def test_encode_and_decode() -> None:
    qubits = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.I(qubits[0]), cirq.measure(qubits[0], key="m"))

    simulator = ft_compiler.FTSimulator()
    result = simulator.run(circuit, repetitions=10)

    np.testing.assert_array_equal(result.measurements["m"], np.zeros((10, 1)))


def test_small_qsvt_circuit() -> None:
    qubit = cirq.LineQubit(0)

    # For phi = pi / 2, exp(i * phi * Z) is logical Z up to global phase. Decompose
    # Z = H X H so that this first QSVT example stays within the supported gate set.
    phase = [cirq.H(qubit), cirq.X(qubit), cirq.H(qubit)]
    u = cirq.X(qubit)
    u_inverse = cirq.inverse(u)
    qsvt_circuit = cirq.Circuit(
        phase,
        u,
        phase,
        u_inverse,
        cirq.measure(qubit, key="m"),
    )

    result = ft_compiler.FTSimulator().run(qsvt_circuit, repetitions=3)

    # Z X Z X = -I, and the global phase does not affect measurement probabilities.
    np.testing.assert_array_equal(result.measurements["m"], np.zeros((3, 1)))


def test_one_qubit_circuit():
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.X(qubits[0]), cirq.CX(qubits[0], qubits[1]), cirq.measure(qubits, key="m"))

    simulator = ft_compiler.FTSimulator()
    result = simulator.run(circuit, repetitions=10)
    print(result)


def test_hello_cirq():
    # Pick a qubit.
    qubits = cirq.LineQubit.range(1)

    # Create a circuit.
    circuit = cirq.Circuit(
    cirq.I(qubits[0]),
    cirq.measure(qubits[0], key='m')
)
    print("Circuit:")
    print(circuit)

    # Simulate the circuit several times.
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=20)
    print("Results:")
    print(result)

@pytest.mark.parametrize("error_index", range(7))
def test_syndrome_and_correct_error(error_index: int) -> None:
    measurement = np.zeros(7, dtype=np.int8)
    measurement[error_index] = 1

    corrected = ft_compiler.correct_error(measurement)

    np.testing.assert_array_equal(corrected, np.zeros(7))
    assert measurement[error_index] == 1  # The input is not mutated.


def test_generate_logical_circuit() -> None:
    data, ancilla0, ancilla1 = (
        cirq.LineQubit.range(start, start + 7) for start in (0, 7, 14)
    )
    circuit = ft_compiler.generate_logical_circuit(
        cirq.X.on_each(*data),
        data[0],
        data,
        ancilla0,
        ancilla1,
        error_probability=0.0,
    )

    result = ft_compiler.qec_simulator(circuit, repetitions=5)

    np.testing.assert_array_equal(result.measurements["teleported_measurements"], np.ones((5, 1)))


def test_validation() -> None:
    with pytest.raises(ValueError, match="must contain 7 qubits"):
        ft_compiler.encode(cirq.LineQubit.range(6))

    blocks = [cirq.LineQubit.range(start, start + 7) for start in (0, 7, 14)]
    with pytest.raises(ValueError, match="must belong to the data block"):
        ft_compiler.generate_logical_circuit([], cirq.LineQubit(100), *blocks)
