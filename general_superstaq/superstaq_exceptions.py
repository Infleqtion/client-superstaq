"""Exceptions for the SuperstaQ API."""

from typing import Optional

import requests


class SuperstaQException(Exception):
    """An exception for errors coming from SuperstaQ's API.

    Attributes:
        status_code: A http status code, if coming from an http response with a failing status.
    """

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(f"Status code: {status_code}, Message: '{message}'")
        self.status_code = status_code
        self.message = message


class SuperstaQModuleNotFoundException(SuperstaQException):
    """
    An exception for SuperstaQ features requiring an uninstalled module."""

    def __init__(self, name: str, context: str):
        message = f"'{context}' requires module '{name}'"
        super().__init__(message)


class SuperstaQNotFoundException(SuperstaQException):
    """An exception for errors from SuperstaQ's API when a resource is not found."""

    def __init__(self, message: str):
        super().__init__(message, status_code=requests.codes.not_found)


class SuperstaQUnsuccessfulJobException(SuperstaQException):
    """An exception for attempting to get info about an unsuccessful job.

    This exception occurs when a job has been canceled, deleted, or failed, and information about
    this job is attempted to be accessed.
    """

    def __init__(self, job_id: str, status: str):
        super().__init__(f"Job {job_id} was {status}.")
