# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# mypy: disable-error-code="no-untyped-def"

# -- Path setup --------------------------------------------------------------
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Superstaq"
copyright = "2024, ColdQuanta, Inc., DBA Infleqtion"
author = "ColdQuanta, Inc., DBA Infleqtion"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.mathjax",  # math rendering in html
    "sphinx.ext.napoleon",  # allows google- and numpy- style docstrings
    "IPython.sphinxext.ipython_console_highlighting",
    "autoapi.extension",
    "sphinx.ext.autodoc",
]

# since our notebooks can involve network I/O (or even costing $), we don't want them to be
# run every time we build the docs. Instead, just use the pre-executed outputs.
nbsphinx_execute = "never"

# In addition, we set the mathjax path to v3, which allows \ket{} (and other commands) to render
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
autosummary_generate = False

templates_path = ["_templates"]

autoapi_dirs = [
    "../../cirq-superstaq/cirq_superstaq",
    "../../general-superstaq/general_superstaq",
    "../../qiskit-superstaq/qiskit_superstaq",
    "../../supermarq-benchmarks/supermarq",
]
autoapi_type = "python"

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

autoapi_ignore = ["*_test.py", "*/checks/*.py", "*conftest.py"]

exclude_patterns = [
    "autoapi/index.rst",
    "apps/supermarq/examples/qre-challenge/grovers-ksat.ipynb",
]

autoapi_member_order = "groupwise"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "logo_only": True,
    "navigation_depth": 5,
}
html_logo = "_static/logos/Superstaq_color.png"
html_css_files = [
    "css/docs-superstaq.css",
]
html_favicon = "_static/logos/Infleqtion_logo.png"


# Replace common aliases (gss, css, qss) in docstrings.
def autodoc_process_docstring(app, what, name, obj, options, lines):
    for i in range(len(lines)):
        lines[i] = lines[i].replace("gss.", "general_superstaq.")
        lines[i] = lines[i].replace("css.", "cirq_superstaq.")
        lines[i] = lines[i].replace("qss.", "qiskit_superstaq.")


def setup(app):
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
