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

    def ibmq_set_token(self, token: str) -> Dict[str, str]:
        """Sets IBMQ token field in database.

        Args:
            token: IBMQ token string.

        Returns:
            JSON dictionary containing status of update (whether or not it failed).
        """
        return self._client.ibmq_set_token({"ibmq_token": token})

    def aqt_upload_configs(
        self,
        pulses: Union[str, Dict[str, Any]],
        variables: Union[str, Dict[str, Any]],
    ) -> Dict[str, str]:
        """Uploads configs for AQT. Arguments can either be dictionaries or paths to valid
        .yaml files. If neither is provided, the existing configuration is preserved.

        Args:
            pulses (optional): dictionary or file path for Pulses calibration data
            variables (optional): dictionary or file path for Variables calibration data
        Returns:
            A dictionary of the status of the update (whether or not it failed)
        """

        if isinstance(pulses, dict) or isinstance(variables, dict):
            try:
                import yaml
            except ImportError:
                raise ModuleNotFoundError(
                    "The PyYAML package is required to upload AQT configurations from dicts. "
                    "You can install it using 'pip install pyyaml'."
                )

        if isinstance(pulses, dict):
            pulses_yaml = yaml.safe_dump(pulses)
        elif isinstance(pulses, str) and os.path.isfile(pulses):
            with open(pulses) as pulses_file:
                pulses_yaml = pulses_file.read()
        else:
            raise ValueError(f"{pulses} is not a dictionary or valid file path")

        if isinstance(variables, dict):
            variables_yaml = yaml.safe_dump(variables)
        elif isinstance(variables, str) and os.path.isfile(variables):
            with open(variables) as variables_file:
                variables_yaml = variables_file.read()
        else:
            raise ValueError(f"{variables} is not a dictionary or valid file path")

        return self._client.aqt_upload_configs({"pulses": pulses_yaml, "variables": variables_yaml})

    def aqt_get_configs(self) -> Dict:
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
