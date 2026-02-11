#!/usr/bin/env python3
# Copyright 2026 Infleqtion
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

from __future__ import annotations

import sys

import checks_superstaq as checks

if __name__ == "__main__":
    args = sys.argv[1:]
    args += ["--exclude", "docs/source/apps/aces/*"]
    args += ["--exclude", "docs/source/apps/dfe/*"]
    args += ["--exclude", "docs/source/apps/supermarq/examples/qre-challenge/*"]
    args += ["--exclude", "docs/source/apps/max_sharpe_ratio_optimization.ipynb"]
    args += ["--exclude", "docs/source/apps/cudaq_logical_aim.ipynb"]
    args += ["--exclude", "docs/source/optimizations/ibm/ibmq_dd.ipynb"]
    args += ["--exclude", "docs/source/optimizations/ibm/ibmq_dd_strategies_qss.ipynb"]
    sys.exit(checks.pytest_.run(*args))
