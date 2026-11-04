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

"""A pytest plugin that measures coverage, labeled by the test file responsible for each line.

``coverage_.py --modular --single-run`` loads this plugin in every pytest process, including each
pytest-xdist worker.  It starts coverage before any ``conftest.py`` file imports the code under
test, and labels every line that runs with a coverage context:

* the empty context while pytest starts, which loads the ``conftest.py`` files of the directory
  that contains every selected test file and of its parents,
* ``<collector node ID>|collect`` while a test file or a directory, with its ``conftest.py``, is
  collected,
* ``<test node ID>|setup``, ``|run``, or ``|teardown`` while a test runs,
* ``import:<path>`` while a measured module is imported after pytest has started, and
* ``unattributed`` at other times, such as between tests and after the last one.

A module is imported once per process, by whichever test file needs it first, but in a process of
its own, the test file of the module would import it too.  So the lines that run while a module is
imported are credited to its own test file, whichever test file imported it.

To make a test file run as it would in a fresh process, the plugin also undoes the state that a
test file leaves behind for the next one on the same worker: after the last test of a test file, it
tears down the session- and package-scoped fixtures that the file used, so that the next test file
sets them up anew, and it empties every ``functools`` cache of the measured modules.  Other state
that a test file leaves in modules is not reset.

Collection is restricted to the selected test files, so that other test files are not imported.
Lines are measured with coverage's C tracer, because ``sys.monitoring`` reports each line only once
and so cannot credit a line to a second test file.  Lines that only a subprocess of a test runs are
not measured.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import pathlib
import sys
from collections.abc import Generator
from typing import TYPE_CHECKING

import coverage
import pytest

if TYPE_CHECKING:
    import types
    from collections.abc import Sequence

    from _pytest.fixtures import FixtureDef

# Environment variables set by coverage_.py.
DATA_FILE_VARIABLE = "CHECKS_SUPERSTAQ_COVERAGE_DATA_FILE"
INCLUDE_VARIABLE = "CHECKS_SUPERSTAQ_COVERAGE_INCLUDE"
TEST_FILES_VARIABLE = "CHECKS_SUPERSTAQ_COVERAGE_TEST_FILES"
BRANCH_VARIABLE = "CHECKS_SUPERSTAQ_COVERAGE_BRANCH"

# The context of lines that run outside of the collection and the tests of any one test file.
UNATTRIBUTED = "unattributed"

# The prefix of the context of lines that run while a module is imported, before its path.
IMPORT_PREFIX = "import:"

# Fixtures of these scopes can outlive the test file that set them up.
SHARED_SCOPES = ("session", "package")


####################################################################################################
# pytest hooks


def pytest_ignore_collect(collection_path: pathlib.Path) -> bool | None:
    """Collect only the selected test files, and the directories that contain them."""
    if _selected_test_files is None:
        return None
    path = _normalize(str(collection_path))
    if collection_path.is_dir():
        prefix = path + os.sep
        return None if any(file.startswith(prefix) for file in _selected_test_files) else True
    return None if path in _selected_test_files else True


@pytest.hookimpl(tryfirst=True)
def pytest_collectstart(collector: pytest.Collector) -> None:
    """Label the lines that run while a test file or directory is collected."""
    _context_stack.append(_current_context[0])
    _switch_context(f"{collector.nodeid}|collect" if collector.nodeid else "")


@pytest.hookimpl(trylast=True)
def pytest_collectreport() -> None:
    """Restore the label of the enclosing collector."""
    if _context_stack:
        _switch_context(_context_stack.pop())


def pytest_collection_finish() -> None:
    """Stop crediting lines to the startup of pytest."""
    _switch_context(UNATTRIBUTED)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Label the setup of a test, and note the shared fixtures that its test file uses."""
    fixture_info = getattr(item, "_fixtureinfo", None)
    for fixturedefs in fixture_info.name2fixturedefs.values() if fixture_info else ():
        for fixturedef in fixturedefs:
            if fixturedef.scope in SHARED_SCOPES:
                _shared_fixtures[id(fixturedef)] = fixturedef
    _switch_context(f"{item.nodeid}|setup")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item: pytest.Item) -> None:
    """Label the body of a test."""
    _switch_context(f"{item.nodeid}|run")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(
    item: pytest.Item, nextitem: pytest.Item | None
) -> Generator[None, object, None]:
    """Label the teardown of a test, and reset shared state after the last test of a test file."""
    _switch_context(f"{item.nodeid}|teardown")
    yield
    try:
        if nextitem is None or _get_test_file(nextitem) != _get_test_file(item):
            _finish_shared_fixtures(item)
            _clear_caches()
    finally:
        _switch_context(UNATTRIBUTED)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish() -> None:
    """Save the coverage data of this process.

    A pytest-xdist worker reports that it has finished after this hook, and may then be stopped, so
    its data is saved here rather than in a later hook.
    """
    if _coverage is not None:
        _coverage.stop()
        _coverage.save()


####################################################################################################
# labels of imports


def get_import_context(path: str) -> str:
    """The context of the lines that run while the module at a path is imported."""
    return IMPORT_PREFIX + _normalize(path)


class _ImportLabeler(importlib.abc.MetaPathFinder):
    """Label the lines that run while a measured module is imported after pytest has started.

    It finds modules with the finders that follow it, and labels only modules that are loaded from
    source files by the standard loader, whose ``exec_module`` it replaces for one import.
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Find a module with the following finders, and label its import if it is measured."""
        finders = sys.meta_path
        start = finders.index(self) + 1 if self in finders else len(finders)
        for finder in finders[start:]:
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if not isinstance(loader, importlib.machinery.SourceFileLoader) or spec.origin is None:
            return spec
        if _normalize(spec.origin) not in _measured_files:
            return spec

        context = get_import_context(spec.origin)
        exec_module = loader.exec_module

        def exec_labeled_module(module: types.ModuleType) -> None:
            previous_context = _current_context[0]
            # Every test file needs the modules that are imported while pytest starts.
            if previous_context:
                _switch_context(context)
            try:
                exec_module(module)
            finally:
                _switch_context(previous_context)
                del loader.exec_module

        # The standard finders create a loader for each module, so this affects one import only.
        loader.exec_module = exec_labeled_module  # type: ignore[method-assign]
        return spec


####################################################################################################
# helper functions and state


def _start_coverage() -> coverage.Coverage | None:
    """Start measuring coverage in this process, if coverage_.py asked for it."""
    data_file = os.environ.get(DATA_FILE_VARIABLE)
    if not data_file:
        return None
    measurement = coverage.Coverage(data_file=data_file, data_suffix=True)
    measurement.set_option("run:core", "ctrace")
    # Measure exactly the files that the check covers, whatever the configuration selects.
    measurement.set_option("run:source", None)
    measurement.set_option("run:source_pkgs", [])
    measurement.set_option("run:include", sorted(_measured_files))
    # Lines that only subprocesses run are not attributed to a test, so they are not measured.
    measurement.set_option("run:patch", [])
    measurement.set_option("run:branch", bool(os.environ.get(BRANCH_VARIABLE)))
    # A pytest-xdist controller runs no tests, so it may measure nothing.
    measurement.set_option("run:disable_warnings", ["no-data-collected"])
    measurement.start()
    return measurement


def _switch_context(context: str) -> None:
    """Label the lines that run from now on."""
    if _coverage is not None:
        _coverage.switch_context(context)
    _current_context[0] = context


def _get_test_file(item: pytest.Item) -> str:
    """The test file of a test."""
    return item.nodeid.split("::")[0]


def _finish_shared_fixtures(item: pytest.Item) -> None:
    """Tear down the shared fixtures that a test file used, so that the next one sets them up anew.

    Tearing down a fixture first tears down the fixtures that depend on it.
    """
    fixturedefs = list(_shared_fixtures.values())
    _shared_fixtures.clear()
    request = getattr(item, "_request", None)
    if request is None:
        return
    for fixturedef in reversed(fixturedefs):
        # FixtureDef.finish uses its request only to name the node in hooks and error messages.
        fixturedef.finish(request)


def _clear_caches() -> None:
    """Empty every functools cache of a function or method defined in a measured module."""
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if not file or _normalize(file) not in _measured_files:
            continue
        for value in list(vars(module).values()):
            candidates = [value]
            if isinstance(value, type) and value.__module__ == module.__name__:
                candidates += [
                    getattr(member, "__func__", member) for member in vars(value).values()
                ]
            for candidate in candidates:
                cache_clear = getattr(candidate, "cache_clear", None)
                if callable(cache_clear):
                    cache_clear()


def _normalize(path: str) -> str:
    """A path in a canonical form, to compare it with others."""
    return os.path.normcase(os.path.realpath(path))


def _read_paths(variable: str) -> set[str] | None:
    """The normalized paths in an environment variable, or None if it is not set."""
    paths = os.environ.get(variable)
    return None if paths is None else {_normalize(path) for path in paths.split(os.pathsep)}


# This module is imported as a pytest plugin before pytest loads any conftest.py file.
_measured_files = _read_paths(INCLUDE_VARIABLE) or set()
_selected_test_files = _read_paths(TEST_FILES_VARIABLE)
_coverage = _start_coverage()
if _coverage is not None:
    sys.meta_path.insert(0, _ImportLabeler())
_current_context = [""]
_context_stack: list[str] = []
_shared_fixtures: dict[int, FixtureDef[object]] = {}
