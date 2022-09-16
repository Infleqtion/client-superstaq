docs-superstaq: Home of documentation for SuperstaQ
===================================================
This repository contains materials that support the [SuperstaQ documentation site]().

## How to update docs-superstaq
TBC

## How to build docs-superstaq locally
### Setup your environment
TBC

### Build the docs
1. `cd` into the `docs` folder
0. `make clean`
0. `make html`
0. `open build/html/index.html`

## How this repository was setup
1. Create repository.
0. Create `.gitignore`, `requirements.txt`, and `.readthedocs.yaml`.
0. Run `git submodule add <GitHub repository URL>` for each submodule you would like to add.
0. Create `docs` folder and `cd docs`. Run `sphinx-quickstart`.
    - Select `y` for `Separate source and build directories`.
    - Enter project and author names. Hit `Enter` on remaining options to select default options.
0. Update `conf.py` in `docs/source`.
0. Run `sphinx_apidoc` with relevant options for modules you want to autodoc.
0. Update `index.rst` in `docs/source`.