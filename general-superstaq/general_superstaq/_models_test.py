# pylint: disable=missing-function-docstring,missing-class-docstring,missing-return-doc,
# pylint: disable=unused-argument
from __future__ import annotations

import datetime

import pydantic
import pytest

import general_superstaq as gss


def test_job_data_list_validation() -> None:
    with pytest.raises(
        pydantic.ValidationError,
        match=(
            "Field input_circuits does not contain the correct number of elements. "
            "Expected 1 but found 2."
        ),
    ):
        gss._models.JobData(
            job_type=gss._models.JobType.DEVICE_SUBMISSION,
            statuses=[gss._models.CircuitStatus.RUNNING],
            status_messages=[None],
            user_email="example@infleqtion.com",
            target="ss_example_qpu",
            provider_id=["example_id"],
            num_circuits=1,
            compiled_circuit_type=gss._models.CircuitType.CIRQ,
            compiled_circuits=["compiled_circuit"],
            input_circuits=["input_circuits", "extra_circuit!"],
            input_circuit_type=gss._models.CircuitType.QISKIT,
            pulse_gate_circuits=[None],
            counts=[None],
            state_vectors=[None],
            results_dicts=[None],
            num_qubits=[3],
            shots=[100],
            dry_run=False,
            submission_timestamp=datetime.datetime(2000, 1, 1, 0, 0, 0),
            last_updated_timestamp=[datetime.datetime(2000, 1, 1, 0, 1, 0)],
            initial_logical_to_physicals=[None],
            final_logical_to_physicals=[None],
        )
