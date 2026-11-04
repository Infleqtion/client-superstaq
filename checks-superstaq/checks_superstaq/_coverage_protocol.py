# Copyright 2026 Infleqtion, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The interface between ``coverage_.py --modular --single-run`` and its pytest plugin.

``coverage_.py`` configures the plugin ``_coverage_contexts``, which runs in every pytest process,
through environment variables, and reads back the coverage contexts that the plugin records.  This
module names both.  It has no side effects, so that importing it does not activate the plugin.
"""

from __future__ import annotations

import os

# Environment variables that coverage_.py sets for the plugin.
DATA_FILE_ENV = "CHECKS_SUPERSTAQ_COVERAGE_DATA_FILE"  # the coverage data file to write
ROOT_DIR_ENV = "CHECKS_SUPERSTAQ_COVERAGE_ROOT_DIR"  # the root directory of the repository
MEASURED_FILES_ENV = "CHECKS_SUPERSTAQ_COVERAGE_MEASURED_FILES"  # paths, separated by os.pathsep
SELECTED_TEST_FILES_ENV = "CHECKS_SUPERSTAQ_COVERAGE_TEST_FILES"  # paths, separated by os.pathsep
BRANCH_ENV = "CHECKS_SUPERSTAQ_COVERAGE_BRANCH"  # nonempty to measure branch coverage

# Coverage contexts that the plugin records, other than those named after pytest nodes.
STARTUP_CONTEXT = "checks-superstaq:startup"
SHARED_IMPORT_CONTEXT = "checks-superstaq:shared-import"
IGNORED_CONTEXT = "checks-superstaq:ignored"

# The phases of a test, each with its own coverage context.
TEST_PHASES = ("setup", "run", "teardown")


def get_collection_context(node_id: str) -> str:
    """Return the coverage context of the collection by a pytest collector."""
    return f"{node_id}|collect"


def get_test_context(node_id: str, phase: str) -> str:
    """Return the coverage context of a phase (in `TEST_PHASES`) of a pytest test."""
    return f"{node_id}|{phase}"


def normalize_path(path: str) -> str:
    """Return a path in a canonical form, to compare it with other paths."""
    return os.path.normcase(os.path.realpath(path))
