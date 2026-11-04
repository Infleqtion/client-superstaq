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

"""A pytest plugin that records the coverage contexts of ``coverage_.py --modular --single-run``.

``coverage_.py --modular --single-run`` runs the selected test files in one pytest run, instead of
one process for each test file, and loads this plugin in every pytest process, including each
pytest-xdist worker.  The plugin starts coverage before pytest loads any ``conftest.py`` file, and
labels the lines that run with coverage contexts, from which ``coverage_.py`` approximates the
coverage that each test file would have in a process of its own (see ``_credited_context_regexes``
in ``coverage_.py``).  The contexts are:

* ``checks-superstaq:startup`` while pytest starts, which includes loading the ``conftest.py`` files
  of the directory that contains every selected test file, and of its parents,
* ``<collector node ID>|collect`` while a pytest collector, such as a directory or a test file,
  collects,
* ``<test node ID>|setup``, ``|run``, or ``|teardown`` during the phases of a test,
* ``checks-superstaq:shared-import`` while a module of the repository, other than a test module,
  runs for the first time in the process after startup, and
* ``checks-superstaq:ignored`` at other times, such as between tests.

Since the test files share processes, the plugin also reduces the state that a test file leaves for
the next one on the same worker.  After the last test of a test file, it finalizes the session- and
package-scoped fixtures that the file's tests used, and calls ``cache_clear`` on the cached
functions that it finds in the modules of the repository.  This does not reproduce a fresh process:
other module state, and side effects that fixture finalizers do not undo, remain.

The plugin sets coverage's run options itself.  It measures exactly the files to check, uses
coverage's C tracer, because coverage's ``sysmon`` core does not support dynamic contexts, and does
not measure subprocesses, whose lines cannot be attributed to a test file.  It also restricts
collection to the selected test files, so that other test files are not imported.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import pathlib
import sys
from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING

import coverage
import pytest

from checks_superstaq import _coverage_protocol as protocol

if TYPE_CHECKING:
    import types

    from _pytest.fixtures import FixtureDef

# Fixtures of these scopes can outlive the test file that set them up.
_SHARED_SCOPES = ("session", "package")


####################################################################################################
# pytest hooks


def pytest_ignore_collect(collection_path: pathlib.Path) -> bool | None:
    """Ignore paths that are neither selected test files nor directories that contain them."""
    if _selected_test_files is None:
        return None
    path = protocol.normalize_path(str(collection_path))
    if collection_path.is_dir():
        prefix = path + os.sep
        is_selected = any(file.startswith(prefix) for file in _selected_test_files)
    else:
        is_selected = path in _selected_test_files
    return None if is_selected else True


@pytest.hookimpl(tryfirst=True)
def pytest_collectstart(collector: pytest.Collector) -> None:
    """Switch to the collection context of a collector (the startup context for the session)."""
    _context_stack.append(_current_context)
    if collector.nodeid:
        _switch_context(protocol.get_collection_context(collector.nodeid))
    else:
        _switch_context(protocol.STARTUP_CONTEXT)


@pytest.hookimpl(trylast=True)
def pytest_collectreport() -> None:
    """Switch back to the context of the enclosing collector."""
    if _context_stack:
        _switch_context(_context_stack.pop())


def pytest_collection_finish() -> None:
    """Switch to the ignored context once collection has finished."""
    _switch_context(protocol.IGNORED_CONTEXT)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Switch to the setup context of a test before its fixtures are set up."""
    _switch_context(protocol.get_test_context(item.nodeid, "setup"))


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item: pytest.Item) -> None:
    """Switch to the run context of a test before its body runs."""
    _switch_context(protocol.get_test_context(item.nodeid, "run"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(
    item: pytest.Item, nextitem: pytest.Item | None
) -> Generator[None, object, None]:
    """Switch to the teardown context of a test, and reduce shared state after its test file."""
    _switch_context(protocol.get_test_context(item.nodeid, "teardown"))
    # The fixtures that the test used, including those that it or its fixtures requested by name.
    request = getattr(item, "_request", None)
    for fixturedef in getattr(request, "_fixture_defs", {}).values():
        if fixturedef.scope in _SHARED_SCOPES:
            _shared_fixtures[id(fixturedef)] = fixturedef
    yield
    try:
        if nextitem is None or _get_test_file(nextitem) != _get_test_file(item):
            _finish_shared_fixtures(request)
            _clear_caches()
    finally:
        _switch_context(protocol.IGNORED_CONTEXT)


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
# shared imports


class _SharedImportLoader(importlib.machinery.SourceFileLoader):
    """A source loader that runs a module in the shared-import context the first time it runs.

    Modules that first run while pytest starts keep the startup context.  A module that runs again,
    as when it is reloaded or imported after its removal from ``sys.modules``, keeps the current
    context, since in a process of its own, a test file would not run it again either.
    """

    def exec_module(self, module: types.ModuleType) -> None:
        """Run a module, in the shared-import context if this is its first run after startup."""
        file = protocol.normalize_path(self.path)
        is_first_run = file not in _executed_files
        _executed_files.add(file)
        if not is_first_run or _current_context == protocol.STARTUP_CONTEXT:
            super().exec_module(module)
            return
        previous_context = _current_context
        _switch_context(protocol.SHARED_IMPORT_CONTEXT)
        try:
            super().exec_module(module)
        finally:
            _switch_context(previous_context)


class _SharedImportFinder(importlib.abc.MetaPathFinder):
    """Load the modules of the repository, other than test modules, with `_SharedImportLoader`.

    Such modules include those that are not measured, such as ``__init__.py`` files, since their
    imports can run measured lines.
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Find a module with the finders after this one, and replace its loader if eligible."""
        finders = sys.meta_path
        start = finders.index(self) + 1 if self in finders else len(finders)
        for finder in finders[start:]:
            find_spec = getattr(finder, "find_spec", None)
            spec = find_spec(fullname, path, target) if find_spec is not None else None
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if (
            type(loader) is importlib.machinery.SourceFileLoader
            and _is_repository_file(protocol.normalize_path(loader.path))
            and not _is_test_file(loader.path)
        ):
            spec.loader = _SharedImportLoader(loader.name, loader.path)
        return spec


####################################################################################################
# helper functions and state


def _start_coverage() -> coverage.Coverage | None:
    """Start coverage in this process, if ``coverage_.py`` configured the plugin."""
    data_file = os.environ.get(protocol.DATA_FILE_ENV)
    if not data_file:
        return None
    measurement = coverage.Coverage(data_file=data_file, data_suffix=True)
    measurement.set_option("run:core", "ctrace")
    measurement.set_option("run:context", None)
    measurement.set_option("run:dynamic_context", None)
    measurement.set_option("run:source", None)
    measurement.set_option("run:source_pkgs", [])
    measurement.set_option("run:include", sorted(_measured_files))
    measurement.set_option("run:patch", [])
    if os.environ.get(protocol.BRANCH_ENV):
        measurement.set_option("run:branch", value=True)
    # A pytest-xdist controller runs no tests, so it may measure nothing.
    measurement.set_option("run:disable_warnings", ["no-data-collected"])
    measurement.start()
    measurement.switch_context(protocol.STARTUP_CONTEXT)
    return measurement


def _switch_context(context: str) -> None:
    """Switch the coverage context of the lines that run from now on."""
    global _current_context  # noqa: PLW0603
    if _coverage is not None:
        _coverage.switch_context(context)
    _current_context = context


def _get_test_file(item: pytest.Item) -> str:
    """Return the file part of the node ID of a test."""
    return item.nodeid.split("::")[0]


def _finish_shared_fixtures(request: pytest.FixtureRequest | None) -> None:
    """Finalize the shared fixtures that a test file used, so that the next one sets them up anew.

    Finalizing a fixture first finalizes the fixtures that depend on it.
    """
    fixturedefs = list(_shared_fixtures.values())
    _shared_fixtures.clear()
    if request is None:
        return
    for fixturedef in reversed(fixturedefs):
        # FixtureDef.finish uses its request only to name the node in hooks and error messages.
        fixturedef.finish(request)  # type: ignore[arg-type]


def _clear_caches() -> None:
    """Call ``cache_clear`` on the cached functions in the modules of the repository.

    The functions are found among the attributes of the modules, and the members of the classes
    that the modules define.  Modules that are not measured are included, since their caches can
    keep measured lines from running, and so are test modules, since another test file can fill the
    cache of a function that a test module defines.
    """
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if not isinstance(file, str) or not _is_repository_file(protocol.normalize_path(file)):
            continue
        for value in list(vars(module).values()):
            candidates = [value]
            if isinstance(value, type) and value.__module__ == module.__name__:
                for member in vars(value).values():
                    is_method = isinstance(member, (classmethod, staticmethod))
                    candidates.append(member.__func__ if is_method else member)
            for candidate in candidates:
                # Look on the type, so that proxies of other objects are not asked.
                if callable(getattr(type(candidate), "cache_clear", None)):
                    candidate.cache_clear()


def _is_repository_file(file: str) -> bool:
    """Whether a normalized path is in the repository and outside ``site-/dist-packages``."""
    parts = set(pathlib.PurePath(file).parts)
    return file.startswith(_root_directory) and not {"site-packages", "dist-packages"} & parts


def _is_test_file(file: str) -> bool:
    """Whether a file is named as a test module or a ``conftest.py`` file."""
    name = os.path.basename(file)
    return name == "conftest.py" or name.endswith("_test.py") or name.startswith("test_")


def _read_paths(variable: str) -> set[str] | None:
    """Return the normalized paths in an environment variable, or None if it is not set."""
    paths = os.environ.get(variable)
    if paths is None:
        return None
    return {protocol.normalize_path(path) for path in paths.split(os.pathsep)}


# This module is imported as a pytest plugin before pytest loads any conftest.py file.
_root_directory = (
    protocol.normalize_path(os.environ.get(protocol.ROOT_DIR_ENV) or os.getcwd()) + os.sep
)
_measured_files = _read_paths(protocol.MEASURED_FILES_ENV) or set()
_selected_test_files = _read_paths(protocol.SELECTED_TEST_FILES_ENV)
_executed_files: set[str] = set()
_shared_fixtures: dict[int, FixtureDef[object]] = {}
_context_stack: list[str] = []
_current_context = protocol.STARTUP_CONTEXT
_coverage = _start_coverage()
if _coverage is not None:
    sys.meta_path.insert(0, _SharedImportFinder())
