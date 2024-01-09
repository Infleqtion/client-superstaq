docs: Home of documentation for Superstaq
===================================================
This repository contains materials that support the [Superstaq documentation site](https://superstaq.readthedocs.io/en/latest/).

## How to build the docs locally
### Setup your environment

Clone the repository, set up your virtual environment, and install requirements.

    git clone git@github.com:Infleqtion/client-superstaq.git
    python3 -m venv venv_superstaq
    source venv_superstaq/bin/activate
    cd client-superstaq/docs
    pip install -r requirements.txt
    

### Build the docs
1. `cd` into the `docs` folder
0. `make clean`
0. `make html`
    Note: After this step you may encounter an error telling you to install pandocs. This means you need to do a systems-level install of pandocs. You can do so by following the directions [here](https://pandoc.org/installing.html). Once done, repeat steps 2-3.
0. `open build/html/index.html`
 
## How to update the docs
1. Make sure you are on the `main` branch in `client-superstaq`.
0. Create a new branch off of `main` in which to make your updates.
0. Make any relevant updates.
0. Push all commits and create a Pull Request.
0. Request the relevant people to review your Pull Request.
0. After your Pull Request has been reviewed, merge in your branch.

## How this repository was setup
1. Create repository.
0. Create `.gitignore`, `requirements.txt`, and `.readthedocs.yaml`.
0. Create `docs` folder and `cd docs`. Run `sphinx-quickstart`.
    - Select `y` for `Separate source and build directories`.
    - Enter project and author names. Hit `Enter` on remaining options to select default options.
0. Update `conf.py` in `docs/source`.
0. Run `sphinx_apidoc` with relevant options for modules you want to autodoc.
0. Update `index.rst` in `docs/source`.
