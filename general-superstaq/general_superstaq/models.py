"""Data models used when communicating with the Server"""

import pydantic
from enum import Enum
import datetime
from typing import Any


class JobType(str, Enum):
    """The different types of jobs that can be submitted through Superstaq."""

    DEVICE_SUBMISSION = "device_submission"
    """A job that involves submitting circuits to live quantum hardware (device)."""
    DEVICE_SIMULATION = "device_simulation"
    """A job that involves simulating a circuit as if it was on some live hardware (device).
    May or may not include noise (depending on options dict)."""
    SIMULATION = "simulation"
    """Simulation of a given quantum circuit but agnostic to any specific device."""
    COMPILE = "compile"
    """A job that only involves compiling a circuit for some given device."""
    CONVERT = "convert"
    """A job that only involves converting circuits between e.g. qiskit and cirq"""


class SimMethod(str, Enum):
    """The different simulation methods that are available."""

    STATEVECTOR = "statevector"
    """Simulation calculates the wavefunction of the state at the end of the circuit."""
    NOISE_SIM = "noise-sim"
    """The simulation applies noise. If used with `device_simulation` this applies realistic device
    noise otherwise the noise needs to be provided by the user."""
    SIM = "sim"
    """Samples the circuits without any noise."""


class CircuitType(str, Enum):
    """The different languages that are recognized by Superstaq."""

    CIRQ = "cirq"
    QISKIT = "qiskit"
    QASM_STRS = "qasm_strs"


class CircuitStatus(str, Enum):
    """The possible statuses of a job/circuit."""

    RECEIVED = "received"
    """The job has been received (and accepted) to the server and is awaiting further action."""
    AWAITING_COMPILE = "awaiting_compile"
    AWAITING_SUBMISSION = "awaiting_submission"
    AWAITING_SIMULATION = "awaiting_simulation"
    AWAITING_CONVERSION = "awaiting_conversion"
    RUNNING = "running"
    """The job is being run by a worker."""
    COMPLETED = "completed"
    """The job is completed."""
    FAILED = "failed"
    """The job failed. A reason should be stored in the job."""
    CANCELLED = "cancelled"
    """The job was cancelled. A reason should be stored in the job."""
    UNRECOGNIZED = "unrecognized"
    """Something has gone wrong! (Treated as terminal)"""
    PENDING = "pending"
    """When a job has been submitted to an external provider but that provider has
    not yet run the job."""
    DELETED = "deleted"
    """When a job has been deleted."""


TERMINAL_CIRCUIT_STATES = [
    CircuitStatus.COMPLETED,
    CircuitStatus.FAILED,
    CircuitStatus.CANCELLED,
    CircuitStatus.UNRECOGNIZED,
    CircuitStatus.DELETED,
]

UNSUCCESSFUL_CIRCUIT_STATES = [
    CircuitStatus.CANCELLED, CircuitStatus.FAILED, CircuitStatus.DELETED,
]


class DefaultPydanticModel(pydantic.BaseModel, use_enum_values=True, extra="forbid"):
    """Default pydantic model used across the web app."""


class JobData(DefaultPydanticModel):
    """A class to store data for a Superstaq job which is returned through to the client."""

    job_type: JobType
    statuses: list[str]
    status_messages: list[str | None]
    user_email: pydantic.EmailStr
    target: str
    num_circuits: int
    compiled_circuit_type: CircuitType
    compiled_circuits: list[str | None]
    input_circuits: list[str]
    input_circuit_type: CircuitType
    counts: list[dict[str, int] | None]
    state_vectors: list[str | None]
    results_dicts: list[str | None]
    dry_run: bool
    submission_timestamp: datetime.datetime
    last_updated_timestamp: list[datetime.datetime | None]


class NewJob(DefaultPydanticModel):
    """The data model for submitting new jobs."""

    job_type: JobType
    target: str
    circuits: str
    circuit_type: CircuitType
    compiled_circuits: str | None = pydantic.Field(default=None)
    compiled_circuit_type: CircuitType | None = pydantic.Field(default=None)
    shots: int = pydantic.Field(default=0, ge=0)
    dry_run: bool = pydantic.Field(default=False)
    sim_method: SimMethod | None = pydantic.Field(default=None, validate_default=True)
    priority: int = pydantic.Field(default=0)
    options_dict: dict[str, Any] = pydantic.Field(default={})
    tags: list[str] = pydantic.Field(default=[])

    @pydantic.field_validator("sim_method")
    def validate_sim_method(
        cls, sim_meth: SimMethod | None, validation_info: pydantic.ValidationInfo
    ) -> SimMethod | None:
        """Validates the `sim_method` argument. If `job_type` is not `DEVICE_SIMULATION` or
        `SIMULATION` then `sim_method` is forced to be None. Otherwise if `sim_method` is None
        this is changed to `SIM`.

        Args:
            sim_meth: The value of the `sim_method` argument.
            validation_info: The validation information.

        Returns:
            The validated value for sim_method
        """
        # If not a simulation job, set sim_method to None
        if validation_info.data["job_type"] not in [
            JobType.SIMULATION,
            JobType.DEVICE_SIMULATION,
        ]:
            return None
        # If sim_method is provided as None, swap this for SIM
        if sim_meth is None:
            return SimMethod.SIM
        # Otherwise return provided value
        return sim_meth
