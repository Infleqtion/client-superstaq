docs-superstaq: Home of documentation for Superstaq
===================================================
This repository contains materials that support the [Superstaq documentation site](https://docs-superstaq.readthedocs.io/).

## How to build docs-superstaq locally
### Setup your environment

Clone the repository, set up your virtual environment, install requirements, and make sure submodules are up-to-date.

    git clone git@github.com:SupertechLabs/docs-superstaq.git
    python3 -m venv docs-superstaq-env
    source docs-superstaq-env/bin/activate
    cd docs-superstaq
    pip install -r requirements.txt
    git submodule update --init --recursive

### Build the docs
1. `cd` into the `docs` folder
0. `make clean`
0. `make html`
0. `open build/html/index.html`

## How to update docs-superstaq
1. Make sure you are on the `main` branch in `docs-superstaq` as well as the client submodules.
0. Make sure submodules are updated with `git pull --recurse-submodules`. Additionally, update relevant dependencies within the submodules (`pip install -e .`, `pip install general-superstaq --upgrade`).
0. Create a new branch off of `main` in which to make your updates.
0. Make any relevant updates.
0. If any updates were made in the client submodules (e.g., `qiskit-superstaq`), run `build_docs.py`.
0. Push all commits and create a Pull Request.
0. Request the relevant people to review your Pull Request.
0. After your Pull Request has been reviewed, merge in your branch.

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
