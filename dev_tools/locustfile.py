"""Locust (https://docs.locust.io/en/stable/what-is-locust.html) is a load testing tool.
To run Locust, set the SUPERSTAQ_API_KEY environment variable and execute
"locust --config=locust.conf" in the "dev_tools" directory. Then
navigate to http://0.0.0.0:8089/, insert the number of users and spawn rate,
then click the "Start swarming" button.
"""

from __future__ import annotations

import logging

import cirq_superstaq as css
import locust.env
import requests


@locust.events.quitting.add_listener
def _(environment: locust.env.Environment) -> None:
    if environment.stats.total.fail_ratio > 0.01:
        logging.error("Test failed due to failure ratio > 1%")
        environment.process_exit_code = 1
    else:
        environment.process_exit_code = 0


SERVICE = css.Service()


class QuickstartUser(locust.HttpUser):
    """Simulates a user during load testing"""

    @locust.task
    def get_targets(self) -> None:
        """Load tests the get_targets endpoint"""
        requests.get = self.client.get
        _ = SERVICE.get_targets()
