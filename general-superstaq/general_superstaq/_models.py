"""Data models used when communicating with the superstaq server."""

# pragma: no cover
from __future__ import annotations

import datetime
import uuid
from collections.abc import Sequence
from enum import Enum
from typing import Any

import pydantic


class JobType(str, Enum):
    """The different types of jobs that can be submitted through Superstaq."""

    SUBMIT = "submit"
    """A job that involves submitting circuits to an external device."""
    SIMULATE = "simulate"
    """A job that requires superstaq to simulate circuits."""
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
    noise otherwise the noise needs to be provided by the user (via the options dict)."""
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
    # AWAITING states - the job is waiting to be processed
    AWAITING_COMPILE = "awaiting_compile"
    """The job is waiting for a worker to compile."""
    AWAITING_SUBMISSION = "awaiting_submission"
    """The job is waiting for a worker to submit the circuit to an external device."""
    AWAITING_SIMULATION = "awaiting_simulation"
    """The job is waiting for a worker to simulate."""
    # Processing states - the job is being handled in some way
    COMPILING = "compiling"
    """The job is being compiled by a worker."""
    RUNNING = "running"
    """The job is currently running on a device. SUBMIT jobs only"""
    SIMULATING = "simulating"
    """The job is currently being simulated. SIMULATE jobs only"""
    PENDING = "pending"
    """When a job has been submitted and is waiting to be run on a QPU. SUBMIT jobs only."""
    # Error states
    FAILED = "failed"
    """The job failed. A reason should be stored in the job."""
    UNRECOGNIZED = "unrecognized"
    """Something has gone wrong! (Treated as terminal)"""
    # Finished states
    COMPLETED = "completed"
    """The job is completed."""
    CANCELLED = "cancelled"
    """The job was cancelled. A reason should be stored in the job."""
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


class DefaultPydanticModel(
    pydantic.BaseModel,
    use_enum_values=True,
    extra="ignore",
    validate_assignment=True,
    validate_default=True,
):
    """Default pydantic model used across the superstaq server."""


class JobData(DefaultPydanticModel):
    """A class to store data for a Superstaq job which is returned through to the client."""

    job_type: JobType
    """The type of job being submitted."""
    statuses: list[CircuitStatus]
    """The current status of each circuit in the job."""
    status_messages: list[str | None]
    """Any status messages for each circuit in the job."""
    user_email: pydantic.EmailStr
    """The email address of the use who submitted the job."""
    target: str
    """The target that the job was submitted to."""
    provider_id: list[str | None]
    """Any provider side ID's for each circuit in the job."""
    num_circuits: int
    """Number of circuits in the job."""
    compiled_circuits: list[str | None]
    """Compiled versions of each input circuits."""
    input_circuits: list[str]
    """The input circuits as serialized strings."""
    circuit_type: CircuitType
    """The circuit type used for representing the circuits."""
    pulse_gate_circuits: list[str | None]
    """Serialized pulse gate circuits (if relevant)."""
    counts: list[dict[str, int] | None]
    """Counts for each input circuit (if available/relevant)."""
    state_vectors: list[str | None]
    """State vector results for each input circuit (if available/relevant)."""
    results_dicts: list[str | None]
    """Serialized results dictionary for each input circuit (if available/relevant)."""
    num_qubits: list[int]
    """Number of qubits required for each circuit."""
    shots: list[int]
    """Number of shots for each circuit."""
    dry_run: bool
    """Flag to indicate a dry-run job."""
    submission_timestamp: datetime.datetime
    """Timestamp when the job was submitted."""
    last_updated_timestamp: list[datetime.datetime | None]
    """Timestamp for when each circuit was last updated."""
    initial_logical_to_physicals: list[str | None]
    """Serialized initial logical-to-physical mapping for each circuit."""
    final_logical_to_physicals: list[str | None]
    """Serialized initial final-to-physical mapping for each circuit."""


class NewJob(DefaultPydanticModel):
    """The data model for submitting new jobs."""

    job_type: JobType
    """The job type."""
    target: str
    """The target."""
    circuits: str
    """Serialized input circuits."""
    circuit_type: CircuitType
    """The input circuit type."""
    verbatim: bool = pydantic.Field(default=False)
    """Whether to skip compile step."""
    shots: int = pydantic.Field(default=0, ge=0)
    """Number of shots."""
    dry_run: bool = pydantic.Field(default=False)
    """Flag for a dry-run."""
    sim_method: SimMethod | None = pydantic.Field(default=None)
    """The simulation method to use. Only used on `simulation` jobs."""
    priority: int = pydantic.Field(default=0)
    """Optional priority level. Note that different roles have their own maximum priority level
    which will limit the priority that users can submit."""
    options_dict: dict[str, Any] = pydantic.Field(default={})
    """Options dictionary with additional configuration detail."""
    tags: list[str] = pydantic.Field(default=[])
    """Optional tags."""


class JobCancellationResults(DefaultPydanticModel):
    """The results from cancelling a job."""

    succeeded: list[str]
    """List of circuits that successfully cancelled."""
    message: str
    """The server message."""
    warnings: list[str]
    """Any warnings generated when cancelling."""


class NewJobResponse(DefaultPydanticModel):
    """Model for the response when a new job is submitted"""

    job_id: uuid.UUID
    """The job ID for the submitted job."""
    num_circuits: int
    """The number of circuits in the job."""


class JobQuery(DefaultPydanticModel):
    """The query model for retrieving jobs. Using multiple values in a field is interpreted as
    logical OR while providing values for multiple fields is interpreted as logical AND."""

    user_email: list[pydantic.EmailStr] | None = pydantic.Field(None)
    """List of user emails to include."""
    job_id: list[uuid.UUID] | None = pydantic.Field(None)
    """List of job IDs to include."""
    target_name: list[str] | None = pydantic.Field(None)
    """List of targets to include."""
    status: list[CircuitStatus] | None = pydantic.Field(None)
    """List of statuses to include."""
    min_priority: int | None = pydantic.Field(None)
    """Minimum priority to include."""
    max_priority: int | None = pydantic.Field(None)
    """Maximum priority to include."""
    submitted_before: datetime.datetime | None = pydantic.Field(None)
    """Filter for jobs submitted before this date."""
    submitted_after: datetime.datetime | None = pydantic.Field(None)
    """Filter for jobs submitted after this date."""


class UserTokenResponse(DefaultPydanticModel):
    """Model for returning a user token to the client, either when adding a new user or
    regenerating the token."""

    email: pydantic.EmailStr
    """The user's email address."""
    token: str
    """The user's new token."""


class BalanceResponse(DefaultPydanticModel):
    """Model for returning a single user balance."""

    email: pydantic.EmailStr
    """The user's email address."""
    balance: float
    """The user's balance."""


class UserInfo(DefaultPydanticModel):
    """Model for the user info returned to the client."""

    name: str
    """User name."""
    email: pydantic.EmailStr
    """User email."""
    role: str
    """User role."""
    balance: float
    """User balance."""
    token: str
    """User API token."""
    user_id: uuid.UUID
    """User id."""


class UserQuery(DefaultPydanticModel):
    """Model for querying the database to retrieve users. Use of lists implied logical OR. Providing
    multiple fields (e.g. name and email) implies logical AND."""

    name: Sequence[str] | None = pydantic.Field(None)
    """List of user names to filter for."""
    email: Sequence[pydantic.EmailStr] | None = pydantic.Field(None)
    """List of user emails to filter for."""
    role: Sequence[str] | None = pydantic.Field(None)
    """List of user roles to filter for."""
    user_id: Sequence[uuid.UUID] | None = pydantic.Field(None)
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
    """New user balance."""


class TargetModel(DefaultPydanticModel):
    """Model for the details of a target."""

    target_name: str
    """The target name."""
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
    """Target is simulator."""
    accessible: bool
    """Target is accessible to user."""


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
    accessible: bool | None = pydantic.Field(None)
    """Include only Superstaq targets that are/aren't accessible to the user."""


class RetrieveTargetInfoModel(DefaultPydanticModel):
    """Model for retrieving detailed target info."""

    target: str
    """The target's name."""
    options_dict: dict[str, Any] = pydantic.Field(dict())
    """The details of the target."""


class TargetInfo(DefaultPydanticModel):
    """Model containing details info about a specific instance of a target."""

    target_info: dict[str, object]
