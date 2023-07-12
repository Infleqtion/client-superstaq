import os
from typing import Any, Dict, Optional, Tuple, Union

import qubovert as qv

import general_superstaq as gss
from general_superstaq import superstaq_client


class Service:
    """This class contains all the services that are used to operate Superstaq."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        remote_host: Optional[str] = None,
        default_target: Optional[str] = None,
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
        """Send acceptance of terms of use at https://superstaq.super.tech/terms_of_use.

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
        maxout: Optional[int] = 1000,
    ) -> Dict[str, str]:
        """Solves the QUBO given via the submit_qubo function in superstaq_client

        Args:
            qubo: A `qv.QUBO` object.
            target: The target to submit the qubo.
            repetitions: Number of repetitions to use. Defaults to 1000.
            method: An optional method parameter. Defaults to None.
            maxout: An optional maxout paramter. Defaults to 1000.

        Returns:
            A dictionary returned by the submit_qubo function.
        """
        return self._client.submit_qubo(qubo, target, repetitions, method, maxout)

    def ibmq_set_token(self, token: str) -> str:
        """Sets IBMQ token field.

        Args:
            token: IBMQ token string.

        Returns:
            String containing status of update (whether or not it failed).
        """
        return self._client.ibmq_set_token({"ibmq_token": token})

    def cq_set_token(self, token: str) -> str:
        """Sets CQ token field.

        Args:
            token: CQ token string.

        Returns:
            String containing status of update (whether or not it failed).
        """
        return self._client.cq_set_token({"cq_token": token})

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
