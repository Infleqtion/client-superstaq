"""Locust (https://docs.locust.io/en/stable/what-is-locust.html) is a load testing tool.
To run Locust, set the SUPERSTAQ_API_KEY environment variable and execute
"locust --config=locust.conf" in the "dev_tools" directory. Then
navigate to http://0.0.0.0:8089/, insert the number of users and spawn rate,
then click the "Start swarming" button.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cirq
import cirq_superstaq
import requests
from locust import HttpUser, events, task

if TYPE_CHECKING:
    from locust.env import Environment


@events.quitting.add_listener
def _(environment: Environment) -> None:
    if environment.stats.total.fail_ratio > 0.01:
        logging.error("Test failed due to failure ratio > 1%")
        environment.process_exit_code = 1
    else:
        environment.process_exit_code = 0


SERVICE = cirq_superstaq.Service()


class QuickstartUser(HttpUser):  # pylint: disable=missing-class-docstring
    @task
    def aqt_compile(self) -> None:  # pylint: disable=missing-function-docstring

        # Construct an example circuit
        q0, q1, q2, q3 = cirq.LineQubit.range(4)
        circuit1 = cirq.Circuit(cirq.H.on_each(q0, q1, q2, q3))

        # Send it to SuperstaQ

        requests.post = self.client.post
        _ = SERVICE.aqt_compile(circuit1)
