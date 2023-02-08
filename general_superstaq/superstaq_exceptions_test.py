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
# pylint: disable=missing-function-docstring

import general_superstaq as gss


def test_superstaq_exception() -> None:
    ex = gss.SuperstaQException(message="Hello", status_code=500)
    assert str(ex) == "Status code: 500, Message: 'Hello'"
    assert ex.status_code == 500
    assert ex.message == "Hello"


def test_module_not_found_exception() -> None:
    ex = gss.SuperstaQModuleNotFoundException("hello_world", "test")
    assert str(ex) == "Status code: None, Message: ''test' requires module 'hello_world''"
    assert ex.message == "'test' requires module 'hello_world'"


def test_superstaq_not_found_exception() -> None:
    ex = gss.SuperstaQNotFoundException(message="Where are you")
    assert str(ex) == "Status code: 404, Message: 'Where are you'"
    assert ex.status_code == 404
    assert ex.message == "Where are you"


def test_superstaq_unsuccessful_job_exception() -> None:
    ex = gss.SuperstaQUnsuccessfulJobException(job_id="SWE", status="canceled")
    assert str(ex) == "Status code: None, Message: 'Job SWE was canceled.'"
    assert ex.status_code is None
    assert ex.message == "Job SWE was canceled."
