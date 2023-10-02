import numbers
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import qubovert as qv

import general_superstaq as gss
from general_superstaq import superstaq_client


class Service:
    """This class contains all the services that are used to operate Superstaq."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        remote_host: Optional[str] = None,
        api_version: str = gss.API_VERSION,
        max_retry_seconds: int = 3600,
        verbose: bool = False,
    ) -> None:
        """Initializes the `Service` class.

        Args:
            client: The Superstaq client to use.
        """

        self._client = superstaq_client._SuperstaqClient(
            client_name="general-superstaq",
            remote_host=remote_host,
            api_key=api_key,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
        )

    def get_balance(self, pretty_output: bool = True) -> Union[str, float]:
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
            return f"${balance:,.2f}"
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

    def submit_qubo(
        self,
        qubo: qv.QUBO,
        target: str,
        repetitions: int = 1000,
        method: Optional[str] = None,
        max_solutions: int = 1000,
    ) -> Dict[str, str]:
        """Solves the QUBO given via the submit_qubo function in superstaq_client, and returns any
        number of specified dictionaries that seek the minimum of the energy landscape from the
        given objective function known as output solutions.

        Args:
            qubo: A `qv.QUBO` object.
            target: The target to submit the qubo.
            repetitions: Number of times that the execution is repeated before stopping.
            method: The parameter specifying method of QUBO solving execution. Currently,
            will either be the "dry-run" option which runs on dwave's simulated annealer,
            or defauls to none and sends it directly to the specified target.
            max_solutions: A parameter that specifies the max number of output solutions.

        Returns:
            A dictionary returned by the submit_qubo function.
        """
        return self._client.submit_qubo(qubo, target, repetitions, method, max_solutions)

    def aqt_upload_configs(self, pulses: Any, variables: Any) -> str:
        """Uploads configs for AQT.

        Arguments can be either file paths (in .yaml format) or qtrl Manager instances.

        Args:
            pulses: PulseManager or file path for pulse configuration.
            variables: VariableManager or file path for variable configuration.

        Returns:
            A status of the update (whether or not it failed).
        """

        def _config_to_yaml_str(config: Any) -> str:
            if isinstance(config, str):
                if not os.path.isfile(config):
                    raise ValueError(f"{config!r} is not a valid file path.")

                with open(config) as config_file:
                    return config_file.read()

            config = getattr(config, "_config_raw", config)  # required to serialize qtrl Managers
            if isinstance(config, dict):
                try:
                    import yaml

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

        pulses_yaml = _config_to_yaml_str(pulses)
        variables_yaml = _config_to_yaml_str(variables)

        return self._client.aqt_upload_configs({"pulses": pulses_yaml, "variables": variables_yaml})

    def aqt_get_configs(self) -> Dict[str, str]:
        """Retrieves the raw AQT config files that had previously been uploaded to Superstaq.

        Returns:
            A dictionary containing all of the user's configs (as YAML strings), indexed by the
            config names (e.g. "pulses", "variables").
        """
        return self._client.aqt_get_configs()

    def aqt_download_configs(
        self,
        pulses_file_path: Optional[str] = None,
        variables_file_path: Optional[str] = None,
        overwrite: bool = False,
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Downloads AQT configs that had previously been uploaded to Superstaq.

        Optionally saves configs to disk as YAML configuration files. Otherwise, the PyYAML package
        is required to read the downloaded configs.

        Args:
            pulses_file_path (optional): Where to write the pulse configuration.
            variables_file_path (optional): Where to write the variable configuration.
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
                print(f"Pulses configuration saved to {pulses_file_path}.")

            with open(variables_file_path, "w") as text_file:
                text_file.write(config_dict["variables"])
                print(f"Variables configuration saved to {variables_file_path}.")

            return None

        elif pulses_file_path or variables_file_path:
            raise ValueError("Please provide both pulses and variables file paths, or neither.")

        else:
            try:
                import yaml
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
        method: Optional[str] = None,
        noise: Optional[str] = None,
        error_prob: Optional[Union[float, Tuple[float, float, float]]] = None,
        tag: Optional[str] = None,
        lifespan: Optional[int] = None,
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
            ValueError: If the target or noise model is not valid.
            SuperstaqServerException: If the request fails.
        """
        noise_dict: Dict[str, object] = {}
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

    def process_aces(self, job_id: str) -> List[float]:
        """Process a job submitted through `submit_aces`.

        Args:
            job_id: The job id returned by `submit_aces`.

        Returns:
            The estimated eigenvalues.
        """
        return self._client.process_aces(job_id=job_id)
