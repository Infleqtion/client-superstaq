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
