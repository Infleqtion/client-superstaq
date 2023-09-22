"""Exceptions for the Superstaq API."""


class SuperstaqException(Exception):
    """An exception for errors coming from Superstaq's API."""

    def __init__(self, message: str) -> None:
        """Initializes the `SupertaqException` class.

        Args:
            message: A message corresponding to the HTTP response status code. Defaults to None.
        """
        self.message = message
        super().__init__(message)


class SuperstaqUnsuccessfulJobException(SuperstaqException):
    """An exception for attempting to get info about an unsuccessful job.

    This exception occurs when a job has been cancelled, deleted, or failed, and information about
    this job is attempted to be accessed.
    """

    def __init__(self, job_id: str, status: str) -> None:
        """Initializes the `SuperstaqUnsuccessfulJobException` class.

        Args:
            job_id: The job identifier of the unsuccessful job.
            status: The status of the unsuccessful job.
        """
        super().__init__(f"Job {job_id} terminated with status {status}.")


class SuperstaqServerException(SuperstaqException):
    """An exception for non-retriable server-side errors."""

    def __init__(self, message: str, status_code: int = 400, contact_info: bool = False) -> None:
        """Initializes the `SuperstaqServerException` class.

        Args:
            message: The message to be displayed for this exception.
            status_code: An HTTP status code, if coming from an HTTP response with a failing
                status.
            contact_info: Whether or not to display contact information.
        """
        status_msg = (
            "400, non-retriable error making request to Superstaq API"
            if (status_code == 400)
            else str(status_code)
        )
        message = f"{message} (Status code: {status_msg})"
        if contact_info:
            slack_invite_url = (
                "https://join.slack.com/t/superstaq/shared_invite/"
                "zt-1wr6eok5j-fMwB7dPEWGG~5S474xGhxw"
            )
            message = (
                f"{message}\n\n"
                "If you would like to contact a member of our team, email us at "
                f"superstaq@infleqtion.com or join our Slack workspace: {slack_invite_url}."
            )

        self.status_code = status_code
        super().__init__(message=message)


class SuperstaqWarning(Warning):
    """A warning issued by the server."""

    def __init__(self, message: str) -> None:  # Overridden to limit to one argument
        super().__init__(message)
