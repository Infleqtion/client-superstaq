import os
from typing import Any, Dict, Optional, Tuple, Union

from general_superstaq import superstaq_client


class UserConfig:
    def __init__(self, client: superstaq_client._SuperstaQClient):
        self._client = client

    def get_balance(self, pretty_output: bool = True) -> Union[str, float]:
        """Get the querying user's account balance in USD.
        Args:
            pretty_output: whether to return a pretty string or a float of the balance.
        Returns:
            If pretty_output is True, returns the balance as a nicely formatted string ($-prefix,
                commas on LHS every three digits, and two digits after period). Otherwise, simply
                returns a float of the balance.
        """
        balance = self._client.get_balance()["balance"]
        if pretty_output:
            return f"${balance:,.2f}"
        return balance

    def ibmq_set_token(self, token: str) -> str:
        """Sets IBMQ token field in database.

        Args:
            token: IBMQ token string.

        Returns:
            JSON dictionary containing status of update (whether or not it failed).
        """
        return self._client.ibmq_set_token({"ibmq_token": token})

    def aqt_upload_configs(self, pulses: Any, variables: Any) -> str:
        """Uploads configs for AQT. Arguments can be either file paths (in .yaml format) or qtrl
        Manager instances.

        Args:
            pulses: PulseManager or file path for Pulses calibration data
            variables: VariableManager or file path for Variables calibration data
        Returns:
            A status of the update (whether or not it failed)
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
        return self._client.aqt_get_configs()

    def aqt_download_configs(
        self,
        pulses_file_path: Optional[str] = None,
        variables_file_path: Optional[str] = None,
        overwrite: bool = False,
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Downloads AQT configs that had previously been uploaded to SuperstaQ, optionally saving
        them to disk. Reading AQT configurations requires the PyYAML package.

        Args:
            pulses_file_path (optional): Where to write the pulse configurations
            variables_file_path (optional): Where to write the variables configurations
            overwrite: Whether or not to overwrite existing files
        Returns (if file paths are not provided):
            pulses: A dictionary containing Pulse configuration data
            variables: A dictionary containing Variables configuration data
        Returns (if file paths are provided):
            None
        Raises:
            ValueError: If either file path already exists and overwrite is not True
            ModuleNotFoundError: If file paths are unspecified and PyYAML cannot be imported
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
                print(f"Pulses configuration save to {pulses_file_path}.")

            with open(variables_file_path, "w") as text_file:
                text_file.write(config_dict["variables"])
                print(f"Variables configuration save to {variables_file_path}.")

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
