"""Exceptions for the Superstaq API."""

from typing import Optional


class SuperstaqException(Exception):
    """An exception for errors coming from Superstaq's API."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        """Initializes the `SupertaqException` class.

        Args:
            status_code: An HTTP status code, if coming from an HTTP response with a failing status.
            message: A message corresponding to the HTTP response status code. Defaults to None.
        """
        self.status_code = status_code
        self.message = message

        if status_code is None:
            err_msg = f"{message}"
        else:
            status_msg = (
                "400, non-retriable error making request to Superstaq API"
                if (status_code == 400)
                else str(status_code)
            )
            err_msg = f"{message} (Status code: {status_msg})"
        super().__init__(err_msg)


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
    """An exception for non-retriable server-side errors.

    This exception is called directly from the backend.
    """

    def __init__(self, message: str, status_code: int = 400) -> None:
        """Initializes the `SuperstaqServerException` class.

        Args:
            message: The message to be displayed for this exception.
        """
        super().__init__(message=message, status_code=status_code)
