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

import textwrap

import general_superstaq as gss


def test_superstaq_exception() -> None:
    ex = gss.SuperstaqServerException(message="Hello.", status_code=500)
    assert str(ex) == "Hello. (Status code: 500)"
    assert ex.status_code == 500
    assert ex.message == "Hello."


def test_superstaq_unsuccessful_job_exception() -> None:
    ex = gss.SuperstaqUnsuccessfulJobException(job_id="SWE", status="Cancelled")
    assert str(ex) == "Job SWE terminated with status Cancelled."
    assert ex.status_code is None
    assert ex.message == "Job SWE terminated with status Cancelled."


def test_superstaq_server_exception() -> None:
    ex = gss.SuperstaqServerException(message="This target only supports terminal measurements.")
    expected = textwrap.fill(
        textwrap.dedent(
            """\
            This target only supports terminal measurements. (Status code: 400, non-retriable error
            making request to Superstaq API)
            """
        ),
        width=120,
    )
    assert str(ex) == expected
    assert ex.status_code == 400
    assert ex.message == "This target only supports terminal measurements."
