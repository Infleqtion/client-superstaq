"""Exceptions for the Superstaq API."""

from typing import Optional

import requests


class SuperstaqException(Exception):
    """An exception for errors coming from Superstaq's API."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        """Initializes the `SupertaqException` class.

        Args:
            status_code: An HTTP status code, if coming from an HTTP response with a failing
                status.
            message: A message corresponding to the HTTP response status code. Defaults to None.
        """
        super().__init__(f"Status code: {status_code}, Message: '{message}'")
        self.status_code = status_code
        self.message = message


class SuperstaqModuleNotFoundException(SuperstaqException):
    """An exception for Superstaq features requiring an uninstalled module."""

    def __init__(self, name: str, context: str):
        """Initializes the `SuperstaqModuleNotFoundException` class.

        Args:
            name: The missing module name.
            context: The context for the exception.
        """
        message = f"'{context}' requires module '{name}'"
        super().__init__(message)


class SuperstaqNotFoundException(SuperstaqException):
    """An exception for errors from Superstaq's API when a resource is not found."""

    def __init__(self, message: str):
        """Intializes the `SuperstaqNotFoundException` class.

        Args:
            message: The message to be displayed for this exception.
        """
        super().__init__(message, status_code=requests.codes.not_found)


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
