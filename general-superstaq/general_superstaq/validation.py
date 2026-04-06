# Copyright 2026 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numbers
import re
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

import general_superstaq as gss

if TYPE_CHECKING:
    import numpy.typing as npt
    from _typeshed import SupportsItems


def validate_integer_param(integer_param: object, min_val: int = 1) -> None:
    """Validates that `integer_param` is an integer and positive (or above a minimum value).

    Args:
        integer_param: The input parameter to validate.
        min_val: Optional parameter to validate if `integer_param` is greater than `min_val`.

    Raises:
        TypeError: If `integer_param` is not an integer.
        ValueError: If `integer_param` is less than `min_val`.
    """
    if not (
        (hasattr(integer_param, "__int__") and int(integer_param) == integer_param)
        or (isinstance(integer_param, str) and integer_param.isdecimal())
    ):
        raise TypeError(f"{integer_param} cannot be safely cast as an integer.")

    if int(integer_param) < min_val:
        raise ValueError(f"{integer_param} is less than the minimum value ({min_val}).")


def validate_bitmap(bitmap: npt.ArrayLike) -> None:
    """Checks that `bitmap` is in an array format acceptable by the Atom picture API.

    Args:
        bitmap: The array-like object to validate.

    Raises:
        TypeError: If `bitmap` is not a two-dimensional array.
        TypeError: If `bitmap` is not a square two-dimensional array.
        ValueError: If `bitmap` contains any values outside of {0, 1, 2}.
    """
    bitmap_array = np.asarray(bitmap)
    if not bitmap_array.ndim == 2:
        raise TypeError("The atom picture `bitmap` must be a 2D array-like object.")
    if not (bitmap_array.shape[0] == bitmap_array.shape[1]):
        raise TypeError("The atom picture `bitmap` must be a square 2D array-like object.")
    if not np.all(np.isin(bitmap_array, [0, 1, 2])):
        raise ValueError("The atom picture `bitmap` must only contain the integers 0, 1, or 2.")


def validate_target(target: str) -> str:
    """Checks that `target` conforms to a valid Superstaq format and device type.

    Args:
        target: A string containing the name of a target device.

    Raises:
        ValueError: If `target` has an invalid format or device type.
    """
    target_device_types = ["qpu", "simulator"]

    # Check valid format
    match = re.fullmatch("^([A-Za-z0-9-]+)_([A-Za-z0-9-.]+)_([a-z]+)", target)
    if not match:
        raise ValueError(
            f"{target!r} does not have a valid string format. Valid target strings should be in "
            "the form '<provider>_<device>_<type>', e.g. 'ibmq_fez_qpu'."
        )

    _, _, device_type = match.groups()

    # Check for valid device type
    if device_type not in target_device_types:
        raise ValueError(
            f"{target!r} does not have a valid target device type. Valid device types are: "
            f"{target_device_types}."
        )

    return target


def get_validated_jaqal_qubits(jaqal_programs: str | Sequence[str]) -> int:
    """Gets the maximum number of qubits that should be initialized for all `jaqal_programs`.

    Args:
        jaqal_programs: The Jaqal program(s) to infer qubit count from.

    Returns:
        The max qubit register size needed for all `jaqal_programs`.

    Raises:
        ValueError: If no qubit count could be inferred from `jaqal_programs`.
    """
    jaqal_programs = [jaqal_programs] if isinstance(jaqal_programs, str) else jaqal_programs
    pattern = re.compile(r"^\s*register\b.*?\[(\d+)\]", re.MULTILINE)
    register_sizes = (int(m.group(1)) for jp in jaqal_programs for m in [pattern.search(jp)] if m)
    inferred_num_qubits = max(register_sizes, default=None)
    if inferred_num_qubits is None:
        raise ValueError("Could not determine number of qubits from Jaqal program register(s).")
    return inferred_num_qubits


def validate_noise_type(noise: dict[str, object], n_qubits: int) -> None:
    """Validates that an ACES noise model is valid.

    Args:
        noise: A noise model parameter.
        n_qubits: Number of qubits the noise model is applied to.

    Raises:
        ValueError: If `noise` is not valid.
    """
    noise_type = noise.get("type")
    if not ((params := noise.get("params")) and isinstance(params, Sequence)):
        raise ValueError("`params` must be a sequence in the dict if `type` is in the dict.")

    if noise_type not in [
        "symmetric_depolarize",
        "bit_flip",
        "phase_flip",
        "asymmetric_depolarize",
    ]:
        raise ValueError(f"{noise_type} is not a valid channel.")

    if noise_type in [
        "bit_flip",
        "phase_flip",
    ]:
        if not (
            len(params) == 1
            and isinstance(params[0], (int, float))
            and params[0] >= 0
            and params[0] <= 1
        ):
            raise ValueError(
                f'{params} must be a single number between 0 and 1 for "bit_flip", and '
                f'"phase_flip".'
            )

    if noise_type == "symmetric_depolarize":
        if not (
            len(params) == 1
            and isinstance(params[0], (int, float))
            and params[0] >= 0
            and params[0] <= (1 / (4**n_qubits - 1))
        ):
            raise ValueError(
                f"{params[0]} must be a single number less than 1 / (4^n - 1) for "
                f'"symmetric_depolarize".'
            )

    if noise_type == "asymmetric_depolarize":
        if not (
            len(params) == 3
            and all(isinstance(v, (int, float)) for v in params)
            and sum(params) <= 1
        ):
            raise ValueError(
                f"{params} must be of the form (p_x, p_y, p_z) such that p_x + p_y + p_z <= 1 "
                f'for "asymmetric_depolarize".'
            )


def validate_qubo(qubo: object) -> None:
    """Validates that the input can be converted into a valid QUBO.

    Args:
        qubo: The input value to validate.

    Raises:
        TypeError: If `qubo` is not a dict-like object.
        TypeError: If the keys of `qubo` are of an invalid type.
        ValueError: If `qubo` contains cubic or further higher degree terms.
        TypeError: If the values in `qubo` are not real numbers.
    """
    if not isinstance(qubo, Mapping):
        raise TypeError("QUBOs must be provided as dict-like objects.")

    for key, val in qubo.items():
        if not isinstance(key, Sequence) or isinstance(key, str):
            raise TypeError(f"{key!r} is not a valid key for a QUBO.")

        if len(key) > 2:
            raise ValueError(f"QUBOs must be quadratic, but key {key!r} has length {len(key)}.")

        if not isinstance(val, numbers.Real):
            raise TypeError("QUBO values must be real numbers.")


def _validate_ibm_channel(ibm_channel: str) -> str:
    if ibm_channel == "ibm_quantum":
        raise ValueError(
            "The 'ibm_quantum' channel has been deprecated and sunset on July 1st, 2025. Instead, "
            "use 'ibm_quantum_platform' (or equivalently, the older 'ibm_cloud') and the "
            "corresponding channel token.",
        )
    if ibm_channel == "ibm_cloud":
        warnings.warn(
            "The 'ibm_cloud' channel will be deprecated in the future. Instead, consider using "
            "'ibm_quantum_platform' (the newer version which points to the same channel and works "
            "interchangeably with the same 'ibm_cloud' token and instance).",
            FutureWarning,
            stacklevel=4,
        )
    elif ibm_channel not in ("ibm_cloud", "ibm_quantum_platform"):
        raise ValueError("`ibmq_channel` must be either 'ibm_cloud' or 'ibm_quantum_platform'.")

    return ibm_channel


def get_validated_qscout_options(
    inferred_num_qubits: int,
    *,
    num_eca_circuits: int | None = None,
    mirror_swaps: bool = False,
    base_entangling_gate: str = "xx",
    num_qubits: int | None = None,
    error_rates: SupportsItems[tuple[int, ...], float] | None = None,
    atol: float = 1e-8,
    atol_map: SupportsItems[tuple[int, ...], float] | None = None,
    keep_qubit_order: bool = False,
    random_seed: int | None = None,
    **kwargs: object,
) -> dict[str, Any]:
    """Generates an options dictionary packaging all keyword args into a format compatible for
        `/qscout_compile` and `jaqal_compile()`.

    Args:
        inferred_num_qubits: The determined number of qubits needed by the endpoint these options
            correspond to.
        num_eca_circuits: Optional number of logically equivalent random circuits for Equivalent
            Circuit Averaging (ECA).
        mirror_swaps: Whether to use mirror swapping to reduce two-qubit gate overhead.
        base_entangling_gate: The base entangling gate to use ("xx", "zz", "sxx", or "szz").
            Compilation with the "xx" and "zz" entangling bases will use arbitrary
            parameterized two-qubit interactions, while the "sxx" and "szz" bases will only use
            fixed maximally-entangling rotations.
        num_qubits: An optional number of qubits that should be initialized in the backend (by
            default this will be determined from `inferred_num_qubits`).
        error_rates: Optional dictionary assigning relative error rates to pairs of physical
            qubits, in the form `{<qubit_indices>: <error_rate>, ...}` where `<qubit_indices>`
            is a tuple physical qubit indices (ints) and `<error_rate>` is a relative error rate
            for gates acting on those qubits (for example `{(0, 1): 0.3, (1, 2): 0.2}`). If
            provided, Superstaq will attempt to map the circuit to minimize the total error on
            each qubit. Omitted qubit pairs are assumed to be error-free.
        atol: Optional tolerance (trace distance bound) used for approximate compilation.
            Superstaq will elide gates which can be approximated within the given tolerance by
            identity operations.
        atol_map: Optional dictionary assigning compilation tolerances to physical qubits, in
            the form `{<qubit_indices>: <atol>, ...}` where `<qubit_indices>` is a tuple of
            physical qubit indices (ints) and `<atol>` is an absolute tolerance (trace distance
            bound) for gates acting on those qubits (for example `{(0, 1): 0.3, (1, 2): 0.2}`).
            If provided, these tolerances will override `atol` for gates on the given qubits.
            Omitted qubit pairs default to `atol`.
        keep_qubit_order: If `True`, do not reorder input qubits when compiling with ECA.
        random_seed: Used to seed any stochastic compilation passes (especially for ECA).
        kwargs: Other desired options.

    Returns:
        The validated options dictionary packaging all `args`.

    Raises:
        ValueError: If `base_entangling_gate` is not a valid gate option.
        ValueError: If provided `num_qubits` is less than the register size determined by
            `inferred_num_qubits`.
    """
    base_entangling_gate = base_entangling_gate.lower()
    if base_entangling_gate not in ("xx", "zz", "sxx", "szz"):
        raise ValueError("`base_entangling_gate` must be 'xx', 'zz', 'sxx', or 'szz'")

    options = {
        "mirror_swaps": mirror_swaps,
        "base_entangling_gate": base_entangling_gate,
        "keep_qubit_order": bool(keep_qubit_order),
        "atol": atol,
        **kwargs,
    }

    if num_eca_circuits is not None:
        gss.validation.validate_integer_param(num_eca_circuits)
        options["num_eca_circuits"] = int(num_eca_circuits)

    if random_seed is not None:
        gss.validation.validate_integer_param(random_seed)
        options["random_seed"] = int(random_seed)

    if error_rates is not None:
        error_rates_list = list(error_rates.items())
        options["error_rates"] = error_rates_list
        inferred_num_qubits = max(
            inferred_num_qubits, *(q + 1 for qs, _ in error_rates_list for q in qs)
        )

    if atol_map is not None:
        atol_map_list = list(atol_map.items())
        options["atol_map"] = atol_map_list
        inferred_num_qubits = max(
            inferred_num_qubits, *(q + 1 for qs, _ in atol_map_list for q in qs)
        )

    # Infer `num_qubits` from inputs, if not already specified
    if num_qubits is None:
        num_qubits = inferred_num_qubits

    gss.validation.validate_integer_param(num_qubits)
    if num_qubits < inferred_num_qubits:
        raise ValueError(f"At least {inferred_num_qubits} qubits are required for this input.")

    options["num_qubits"] = num_qubits
    return options
