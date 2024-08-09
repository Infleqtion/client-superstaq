# Copyright 2024 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Locust (https://docs.locust.io/en/stable/what-is-locust.html) is a load testing tool.
To run Locust, first install it with "python -m pip install locust", set the SUPERSTAQ_API_KEY
environment variable and execute "locust --config=locust.conf" in the "dev_tools" directory.
Navigate to http://0.0.0.0:8089/ in your browser, insert the number of users and spawn rate,
and then click the "Start swarming" button."""

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
