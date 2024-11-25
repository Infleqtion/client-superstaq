"""Data models used when communicating with the Server"""

from __future__ import annotations

# pragma: no cover
import datetime
import uuid
from enum import Enum
from typing import Any

import pydantic


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
    CircuitStatus.CANCELLED,
    CircuitStatus.FAILED,
    CircuitStatus.DELETED,
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
    provider_id: list[str | None]
    num_circuits: int
    compiled_circuit_type: CircuitType
    compiled_circuits: list[str | None]
    input_circuits: list[str]
    input_circuit_type: CircuitType
    pulse_gate_circuits: list[str | None]
    counts: list[dict[str, int] | None]
    state_vectors: list[str | None]
    results_dicts: list[str | None]
    num_qubits: list[int]
    shots: list[int]
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


class JobCancellationResults(DefaultPydanticModel):
    """The results from cancelling a job."""

    succeeded: list[str]
    message: str
    warnings: list[str]


class NewJobResponse(DefaultPydanticModel):
    """Model for the response when a new job is submitted"""

    job_id: uuid.UUID
    num_circuits: int


class JobQuery(DefaultPydanticModel):
    """The query model for retrieving jobs. Using multiple values in a field is interpreted as
    logical OR while providing values for multiple fields is interpreted as logical AND."""

    user_email: list[pydantic.EmailStr] | None = pydantic.Field(None)
    job_id: list[uuid.UUID] | None = pydantic.Field(None)
    target_name: list[str] | None = pydantic.Field(None)
    status: list[CircuitStatus] | None = pydantic.Field(None)
    min_priority: int | None = pydantic.Field(None)
    max_priority: int | None = pydantic.Field(None)
    submitted_before: datetime.datetime | None = pydantic.Field(None)
    submitted_after: datetime.datetime | None = pydantic.Field(None)


class UserTokenResponse(DefaultPydanticModel):
    """Model for returning a user token to the client, either when adding a new user or
    regenerating the token."""

    email: pydantic.EmailStr
    token: str


class BalanceResponse(DefaultPydanticModel):
    """Model for returning a single user balance."""

    email: pydantic.EmailStr
    balance: float


class UserInfo(DefaultPydanticModel):
    """Model for the user info returned to the client."""

    name: str
    """User name."""
    email: pydantic.EmailStr
    """User email."""
    role: str
    """User role."""
    balance: float
    """User balance"""
    token: str
    """User API token."""
    user_id: uuid.UUID
    """User id"""


class UserQuery(DefaultPydanticModel):
    """Model for querying the database to retrieve users. Use of lists implied logical OR. Providing
    multiple fields (e.g. name and email) implies logical AND."""

    name: list[str] | None = pydantic.Field(None)
    """List of user names to filter for."""
    email: list[pydantic.EmailStr] | None = pydantic.Field(None)
    """List of user emails to filter for."""
    role: list[str] | None = pydantic.Field(None)
    """List of user roles to filter for."""
    user_id: list[uuid.UUID] | None = pydantic.Field(None)
    """List of explicit user IDs to filter for."""


class AQTConfigs(DefaultPydanticModel):
    """Model for AQT configs."""

    pulses: str
    """The serialized pulses."""
    variables: str
    """The serialized variables."""


class NewUser(DefaultPydanticModel):
    """Model for creating new users."""

    name: str
    """User name."""
    email: pydantic.EmailStr
    """User email."""
    role: str | None = pydantic.Field(None)
    """User role."""
    initial_balance: float | None = pydantic.Field(None)
    """Initial balance."""


class UpdateUserDetails(DefaultPydanticModel):
    """Model for requests which modify user details"""

    name: str | None = pydantic.Field(None)
    """New user name."""
    role: str | None = pydantic.Field(None)
    """New user role."""
    balance: float | None = pydantic.Field(None)
    """New user balance"""


class TargetInfoModel(DefaultPydanticModel):
    """Model for the info of a target."""

    target_name: str
    """The target name"""
    supports_submit: bool
    """Targets allow job submission."""
    supports_submit_qubo: bool
    """Targets allows QUBO submission."""
    supports_compile: bool
    """Target allows circuit compilation."""
    available: bool
    """Target is currently available."""
    retired: bool
    """Target is retired."""
    simulator: bool
    """Target is simulator"""


class GetTargetsFilterModel(DefaultPydanticModel):
    """Model for /get_target requests."""

    simulator: bool | None = pydantic.Field(None)
    """Include Superstaq targets that are/not simulators."""
    supports_submit: bool | None = pydantic.Field(None)
    """Include Superstaq targets that allow/do not allow job submission."""
    supports_submit_qubo: bool | None = pydantic.Field(None)
    """Include Superstaq targets that allow/do not allow QUBO submission."""
    supports_compile: bool | None = pydantic.Field(None)
    """Include Superstaq targets that allow/do not allow circuit compilation."""
    available: bool | None = pydantic.Field(True)
    """Include Superstaq targets that are/not currently available."""
    retired: bool = pydantic.Field(False)
    """Include Superstaq targets that are retired."""


class RetrieveTargetInfoModel(DefaultPydanticModel):
    """Model for retrieving target info."""

    target: str
    options_dict: dict[str, Any] = pydantic.Field(dict())
