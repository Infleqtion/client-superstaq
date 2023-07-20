# Copyright 2021 The Cirq Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=missing-function-docstring,missing-class-docstring

import general_superstaq as gss


def test_superstaq_exception() -> None:
    ex = gss.SuperstaqServerException(message="Hello.", status_code=500)
    assert str(ex) == "Hello. (Status code: 500)"
    assert ex.message == "Hello. (Status code: 500)"
    assert ex.status_code == 500


def test_superstaq_unsuccessful_job_exception() -> None:
    ex = gss.SuperstaqUnsuccessfulJobException(job_id="SWE", status="Cancelled")
    assert str(ex) == "Job SWE terminated with status Cancelled."
    assert ex.message == "Job SWE terminated with status Cancelled."


def test_superstaq_server_exception() -> None:
    ex = gss.SuperstaqServerException(message="This target only supports terminal measurements.")
    expected = (
        "This target only supports terminal measurements. (Status code: 400, non-retriable error "
        "making request to Superstaq API)"
    )
    assert str(ex) == expected
    assert ex.message == expected
    assert ex.status_code == 400


def test_exception_with_contact_info() -> None:
    ex = gss.SuperstaqServerException(
        message="qtm_lt-s8_qpu is not an available Quantinuum device.",
        status_code=400,
        contact_info=True,
    )
    slack_invite_url = (
        "https://join.slack.com/t/superstaq/shared_invite/zt-1wr6eok5j-fMwB7dPEWGG~5S474xGhxw"
    )
    expected = (
        "qtm_lt-s8_qpu is not an available Quantinuum device. (Status code: 400, non-retriable "
        "error making request to Superstaq API)\n\n"
        "If you would like to contact a member of our team, email us at "
        f"superstaq@infleqtion.com or join our Slack workspace: {slack_invite_url}."
    )
    assert str(ex) == expected
    assert ex.message == expected
    assert ex.status_code == 400
