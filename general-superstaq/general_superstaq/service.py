# Copyright 2025 Infleqtion
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
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, overload

import general_superstaq as gss
from general_superstaq.superstaq_client import _SuperstaqClient, _SuperstaqClientV3

if TYPE_CHECKING:
    import numpy.typing as npt

TQuboKey = TypeVar("TQuboKey")

CLIENT_VERSION = {
    "v0.2.0": _SuperstaqClient,
    "v0.3.0": _SuperstaqClientV3,
}


class Service:
    """This class contains all the services that are used to operate Superstaq."""

    def __init__(
        self,
        api_key: str | None = None,
        remote_host: str | None = None,
        api_version: str = gss.API_VERSION,
        max_retry_seconds: int = 3600,
        verbose: bool = False,
    ) -> None:
        """Initializes the `Service` class.

        Args:
            api_key: The key used for authenticating against the Superstaq API.
            remote_host: The url of the server exposing the Superstaq API. This will strip anything
                besides the base scheme and netloc, i.e. it only takes the part of the host of
                the form `http://example.com` of `http://example.com/test`.
            api_version: Which version of the API to use. Defaults to `client_superstaq.API_VERSION`
                (which is the most recent version when this client was downloaded).
            max_retry_seconds: The time to continue retriable responses. Defaults to 3600.
            verbose: Whether to print to stderr and stdio any retriable errors that are encountered.
        """
        client_version = CLIENT_VERSION[api_version]
        self._client = client_version(
            client_name="general-superstaq",
            remote_host=remote_host,
            api_key=api_key,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
        )

    def get_balance(self, pretty_output: bool = True) -> str | float:
        """Get the querying user's account balance in USD.

        Args:
            pretty_output: Whether to return a pretty string or a float of the balance.

        Returns:
            If `pretty_output` is `True`, returns the balance as a nicely formatted string
            ($-prefix, commas on LHS every three digits, and two digits after period).
            Otherwise, simply returns a float of the balance.
        """
        balance = self._client.get_balance()["balance"]
        if pretty_output:
            return f"{balance:,.2f} credits"
        return balance

    def _accept_terms_of_use(self, user_input: str) -> str:
        """Send acceptance of terms of use at https://superstaq.infleqtion.com/terms_of_use.

        Args:
            user_input: If "YES", server will mark user as having accepted terms of use.

        Returns:
            A string message indicating if user has been marked as having accepted terms of use.
        """
        return self._client._accept_terms_of_use(user_input)

    def add_new_user(self, name: str, email: str) -> str:
        """Adds new user.

        Args:
            name: The name to add.
            email: The new user's email.

        Returns:
            String containing status of update (whether or not it failed) and the new user's token.
        """
        return self._client.add_new_user(
            {
                "name": name,
                "email": email,
            }
        )

    def update_user_balance(self, email: str, balance: float) -> str:
        """Updates user's balance.

        Args:
            email: The new user's email.
            balance: The new balance.

        Raises:
            superstaq.SuperstaqException: If requested balance exceeds the limit.

        Returns:
            String containing status of update (whether or not it failed).
        """
        limit = 2000.0  # If limit modified, must update in server-superstaq
        if balance > limit:
            raise gss.SuperstaqException(
                f"Requested balance {balance} exceeds limit of {limit}.",
            )
        return self._client.update_user_balance(
            {
                "email": email,
                "balance": balance,
            }
        )

    def update_user_role(self, email: str, role: int) -> str:
        """Updates user's role.

        Args:
            email: The new user's email.
            role: The new role.

        Returns:
            String containing status of update (whether or not it failed).
        """
        return self._client.update_user_role(
            {
                "email": email,
                "role": role,
            }
        )

    def get_targets(
        self,
        simulator: bool | None = None,
        supports_submit: bool | None = None,
        supports_submit_qubo: bool | None = None,
        supports_compile: bool | None = None,
        available: bool | None = None,
        retired: bool | None = None,
        accessible: bool | None = None,
        **kwargs: bool,
    ) -> list[gss.Target]:
        """Gets a list of Superstaq targets along with their status information.

        Args:
            simulator: Optional flag to restrict the list of targets to (non-) simulators.
            supports_submit: Optional boolean flag to only return targets that (don't) allow
                circuit submissions.
            supports_submit_qubo: Optional boolean flag to only return targets that (don't)
                allow qubo submissions.
            supports_compile: Optional boolean flag to return targets that (don't) support
                circuit compilation.
            available: Optional boolean flag to only return targets that are (not) available
                to use.
            retired: Optional boolean flag to only return targets that are or are not retired.
            accessible: Optional boolean flag to only return targets that are (not) accessible
                to the user.
            kwargs: Any additional, supported flags to restrict/filter returned targets.

        Returns:
            A list of Superstaq targets matching all provided criteria.
        """
        filters = dict(
            simulator=simulator,
            supports_submit=supports_submit,
            supports_submit_qubo=supports_submit_qubo,
            supports_compile=supports_compile,
            available=available,
            retired=retired,
            accessible=accessible,
            **kwargs,
        )
        return self._client.get_targets(**filters)

    def get_my_targets(self) -> list[gss.Target]:
        """Gets a list of Superstaq targets that the user can submit to and are available along
        with their status information.

        Returns:
            A list of Superstaq targets that the user can currently submit to.
        """
        return self._client.get_my_targets()

    @overload
    def get_user_info(self) -> dict[str, str | float]: ...

    @overload
    def get_user_info(self, *, name: str) -> list[dict[str, str | float]]: ...

    @overload
    def get_user_info(self, *, email: str) -> list[dict[str, str | float]]: ...

    @overload
    def get_user_info(self, *, user_id: int) -> list[dict[str, str | float]]: ...

    @overload
    def get_user_info(self, *, name: str, user_id: int) -> list[dict[str, str | float]]: ...

    @overload
    def get_user_info(self, *, email: str, user_id: int) -> list[dict[str, str | float]]: ...

    @overload
    def get_user_info(self, *, name: str, email: str) -> list[dict[str, str | float]]: ...

    @overload
    def get_user_info(
        self, *, name: str, email: str, user_id: int
    ) -> list[dict[str, str | float]]: ...

    def get_user_info(
        self,
        *,
        name: str | None = None,
        email: str | None = None,
        user_id: int | None = None,
    ) -> dict[str, str | float] | list[dict[str, str | float]]:
        """Gets a dictionary of the user's info.

        .. note::

            SUPERTECH users can submit optional :code:`name` and/or :code:`email` keyword only
            arguments which can be used to search for the info of arbitrary users on the server.

        Args:
            name: A name to search by. Defaults to None.
            email: An email address to search by. Defaults to None.
            user_id: A user ID to search by. Defaults to None.

        Returns:
            A dictionary of the user information. In the case that either the name or email
            query kwarg is used, a list of dictionaries is returned, corresponding to the user
            information for each user that matches the query.
        """
        user_info = self._client.get_user_info(name=name, email=email, user_id=user_id)

        if name is None and email is None and user_id is None:
            # If no query then return the only element in the list.
            return user_info[0]

        return user_info

    def submit_qubo(
        self,
        qubo: Mapping[tuple[TQuboKey, ...], float],
        target: str = "ss_unconstrained_simulator",
        repetitions: int = 10,
        method: str = "sim_anneal",
        max_solutions: int = 1000,
        *,
        qaoa_depth: int = 1,
        rqaoa_cutoff: int = 0,
        dry_run: bool = False,
        random_seed: int | None = None,
        **kwargs: object,
    ) -> list[dict[TQuboKey, int]]:
        """Solves a submitted QUBO problem via annealing.

        This method returns any number of specified dictionaries that seek the minimum of
        the energy landscape from the given objective function known as output solutions.

        Args:
            qubo: A dictionary representing the QUBO object. The tuple keys represent the
                boolean variables of the QUBO and the values represent the coefficients.
                As an example, for a QUBO with integer coefficients = 2*a + a*b - 5*b*c - 3
                (where a, b, and c are boolean variables), the corresponding dictionary format
                would be {('a',): 2, ('a', 'b'): 1, ('b', 'c'): -5, (): -3}.
            target: The target to submit the QUBO.
            repetitions: Number of times that the execution is repeated before stopping.
            method: The parameter specifying method of QUBO solving execution. Currently, the
                supported methods include "bruteforce", "sim_anneal", "qaoa", or "rqaoa".
                Defaults to "sim_anneal" which runs on DWave's simulated annealer.
            max_solutions: A parameter that specifies the max number of output solutions.
            qaoa_depth: The number of QAOA layers to use. Defaults to 1.
            rqaoa_cutoff: The stopping point for RQAOA before using switching to a classical
                solver. Defaults to 0.
            dry_run: If `method="qaoa"`, a boolean flag to (not) run an ideal 'dry-run'
                QAOA execution on `target`.
            random_seed: Optional random seed choice for RQAOA.
            kwargs: Any optional keyword arguments supported by the qubo solver methods.

        Returns:
            A dictionary containing the output solutions.
        """
        result_dict = self._client.submit_qubo(
            qubo,
            target,
            repetitions,
            method,
            max_solutions,
            qaoa_depth=qaoa_depth,
            rqaoa_cutoff=rqaoa_cutoff,
            dry_run=dry_run,
            random_seed=random_seed,
            **kwargs,
        )
        return gss.serialization.deserialize(result_dict["solution"])

    @staticmethod
    def _qtrl_config_to_yaml_str(config: object) -> str:
        if isinstance(config, str):
            if not os.path.isfile(config):
                raise ValueError(f"{config!r} is not a valid file path.")

            with open(config) as config_file:
                return config_file.read()

        config = getattr(config, "_config_raw", config)  # required to serialize qtrl Managers
        if isinstance(config, dict):
            try:
                import yaml  # noqa: PLC0415

                return yaml.safe_dump(config)

            except ImportError:
                raise ModuleNotFoundError(
                    "The PyYAML package is required to upload AQT configurations from dicts. "
                    "You can install it using 'pip install pyyaml'."
                )
            except yaml.YAMLError:
                pass

        raise ValueError(
            "Unable to serialize configuration. AQT configs should be qtrl Manager instances "
            "or valid file paths."
        )

    def aqt_upload_configs(self, pulses: object, variables: object) -> str:
        """Uploads configs for AQT.

        Arguments can be either file paths (in .yaml format) or qtrl Manager instances.

        Args:
            pulses: `PulseManager` or file path for pulse configuration.
            variables: `VariableManager` or file path for variable configuration.

        Returns:
            A status of the update (whether or not it failed).
        """
        pulses_yaml = self._qtrl_config_to_yaml_str(pulses)
        variables_yaml = self._qtrl_config_to_yaml_str(variables)

        return self._client.aqt_upload_configs({"pulses": pulses_yaml, "variables": variables_yaml})

    def aqt_get_configs(self) -> dict[str, str]:
        """Retrieves the raw AQT config files that had previously been uploaded to Superstaq.

        Returns:
            A dictionary containing all of the user's configs (as YAML strings), indexed by the
            config names (e.g. "pulses", "variables").
        """
        return self._client.aqt_get_configs()

    def aqt_download_configs(
        self,
        pulses_file_path: str | None = None,
        variables_file_path: str | None = None,
        overwrite: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Downloads AQT configs that had previously been uploaded to Superstaq.

        Optionally saves configs to disk as YAML configuration files. Otherwise, the PyYAML package
        is required to read the downloaded configs.

        Args:
            pulses_file_path: Where to write the pulse configuration.
            variables_file_path: Where to write the variable configuration.
            overwrite: Whether or not to overwrite existing files.

        Returns:
            (If file paths are not provided) a tuple of pulses (a dictionary containing pulse
            configuration data) and variables (a dictionary containing calibration variables).

        Raises:
            ValueError: If either file path already exists and overwrite is not True.
            ModuleNotFoundError: If file paths are unspecified and PyYAML cannot be imported.
        """
        if pulses_file_path and variables_file_path:
            pulses_file_exists = os.path.exists(pulses_file_path)
            variables_file_exists = os.path.exists(variables_file_path)

            if not overwrite and pulses_file_exists and variables_file_exists:
                raise ValueError(
                    f"{pulses_file_path} and {variables_file_path} exist. Please try different "
                    "filenames to write to, or pass overwrite=True to overwrite the existing files."
                )
            elif not overwrite and pulses_file_exists:
                raise ValueError(
                    f"{pulses_file_path} exists. Please try a different filename to write to, "
                    "or pass overwrite=True to overwrite the existing file."
                )
            elif not overwrite and variables_file_exists:
                raise ValueError(
                    f"{variables_file_path} exists Please try a different filename to write to, "
                    "or pass overwrite=True to overwrite the existing file."
                )

            config_dict = self.aqt_get_configs()
            with open(pulses_file_path, "w") as text_file:
                text_file.write(config_dict["pulses"])
                print(f"Pulses configuration saved to {pulses_file_path}.")  # noqa: T201

            with open(variables_file_path, "w") as text_file:
                text_file.write(config_dict["variables"])
                print(f"Variables configuration saved to {variables_file_path}.")  # noqa: T201

            return None

        elif pulses_file_path or variables_file_path:
            raise ValueError("Please provide both pulses and variables file paths, or neither.")

        else:
            try:
                import yaml  # noqa: PLC0415
            except ImportError:
                raise ModuleNotFoundError(
                    "The PyYAML package is required to parse AQT configuration files. "
                    "You can install it using 'pip install pyyaml'."
                )

            config_dict = self.aqt_get_configs()
            pulses = yaml.safe_load(config_dict["pulses"])
            variables = yaml.safe_load(config_dict["variables"])

            return pulses, variables

    def submit_aces(
        self,
        target: str,
        qubits: Sequence[int],
        shots: int,
        num_circuits: int,
        mirror_depth: int,
        extra_depth: int,
        method: str | None = None,
        noise: str | None = None,
        error_prob: float | tuple[float, float, float] | None = None,
        tag: str | None = None,
        lifespan: int | None = None,
    ) -> str:
        """Submits the jobs to characterize `target` through the ACES protocol.

        The following gate eigenvalues are eestimated. For each qubit in the device, we consider
        six Clifford gates. These are given by the XZ maps: XZ, ZX, -YZ, -XY, ZY, YX. For each of
        these gates, three eigenvalues are returned (X, Y, Z, in that order). Then, the two-qubit
        gate considered here is the CZ in linear connectivity (each qubit n with n + 1). For this
        gate, 15 eigenvalues are considered: XX, XY, XZ, XI, YX, YY, YZ, YI, ZX, ZY, ZZ, ZI, IX, IY
        IZ, in that order.

        If n qubits are characterized, the first 18 * n entries of the list returned by
        `process_aces` will contain the  single-qubit eigenvalues for each gate in the order above.
        After all the single-qubit eigenvalues, the next 15 * (n - 1) entries will contain for the
        CZ connections, in ascending order.

        The protocol in detail can be found in: https://arxiv.org/abs/2108.05803.

        Args:
            target: The device target to characterize.
            qubits: A list with the qubit indices to characterize.
            shots: How many shots to use per circuit submitted.
            num_circuits: How many random circuits to use in the protocol.
            mirror_depth: The half-depth of the mirror portion of the random circuits.
            extra_depth: The depth of the fully random portion of the random circuits.
            method: Which type of method to execute the circuits with.
            noise: Noise model to simulate the protocol with. Valid strings are
                "symmetric_depolarize", "phase_flip", "bit_flip" and "asymmetric_depolarize".
            error_prob: The error probabilities if a string was passed to `noise`.
                * For "asymmetric_depolarize", `error_prob` will be a three-tuple with the error
                rates for the X, Y, Z gates in that order. So, a valid argument would be
                `error_prob = (0.1, 0.1, 0.1)`. Notice that these values must add up to less than
                or equal to 1.
                * For the other channels, `error_prob` is one number less than or equal to 1, e.g.,
                `error_prob = 0.1`.
            tag: Tag for all jobs submitted for this protocol.
            lifespan: How long to store the jobs submitted for in days (only works with right
                permissions).

        Returns:
            A string with the job id for the ACES job created.

        Raises:
            ValueError: If the target or noise model are not valid.
            ~gss.SuperstaqServerException: If the request fails.
        """
        noise_dict: dict[str, object] = {}
        if noise:
            noise_dict["type"] = noise
            noise_dict["params"] = (
                (error_prob,) if isinstance(error_prob, numbers.Number) else error_prob
            )

        return self._client.submit_aces(
            target=target,
            qubits=qubits,
            shots=shots,
            num_circuits=num_circuits,
            mirror_depth=mirror_depth,
            extra_depth=extra_depth,
            method=method,
            noise=noise_dict,
            tag=tag,
            lifespan=lifespan,
        )

    def process_aces(self, job_id: str) -> list[float]:
        """Process a job submitted through `submit_aces`.

        Args:
            job_id: The job id returned by `submit_aces`.

        Returns:
            The estimated eigenvalues.
        """
        return self._client.process_aces(job_id=job_id)

    def submit_atom_picture(self, bitmap: npt.ArrayLike) -> str:
        """Performs a POST request on the `/atom_picture` endpoint.

        Args:
            bitmap: A 2D array-like object of integers from the set {0, 1, 2}. '0' is empty,
                '1' is atom, and '2' is whatever is there.

        Returns:
            A string containing the post request id.
        """
        request_id = self._client.submit_atom_picture(bitmap=bitmap).get("request_id")
        return f"Submitted request for atom picture with ID: {request_id}"
