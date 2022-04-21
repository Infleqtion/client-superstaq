import os
from typing import Dict, Union


from applications_superstaq import superstaq_client


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

    def aqt_upload_configs(self, pulses_file_path: str, variables_file_path: str) -> Dict[str, str]:
        """Uploads configs for AQT
        Args:
            pulses_file_path: The filepath for Pulses.yaml
            variables_file_path: The filepath for Variables.yaml
        Returns:
            A dictionary of of the status of the update (Whether or not it failed)
        """
        with open(pulses_file_path) as pulses_file:
            read_pulses = pulses_file.read()

        with open(variables_file_path) as variables_file:
            read_variables = variables_file.read()

        json_dict = self._client.aqt_upload_configs(
            {"pulses": read_pulses, "variables": read_variables}
        )

        return json_dict

    def aqt_get_configs(self) -> Dict:
        return self._client.aqt_get_configs()

    def aqt_save_configs(
        self, pulses_file_path: str, variables_file_path: str, overwrite: bool = False
    ) -> None:
        """Writes AQT configs from the AQT system onto the given file paths.

        Args:
            pulses_file_path: Where to write the pulse configurations
            variables_file_path: Where to write the variables configurations
            overwrite: Whether or not to overwrite existing files
        Returns:
            None
        Raises:
            ValueError: If either file path already exists and overwrite is not True
        """
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

        with open(variables_file_path, "w") as text_file:
            text_file.write(config_dict["variables"])
