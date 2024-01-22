docs: Home of documentation for Superstaq
===================================================
This repository contains materials that support the [Superstaq documentation site](https://superstaq.readthedocs.io/en/latest/).

## How to build the docs locally
### Setup your environment

Clone the repository and set up your virtual environment

    git clone git@github.com:Infleqtion/client-superstaq.git
    python3 -m venv venv_superstaq
    source venv_superstaq/bin/activate
    python3 -m pip install -e ./checks-superstaq -e ./general-superstaq -e ./qiskit-superstaq -e ./cirq-superstaq -e ./supermarq-benchmarks 
    Note: Packages should be installed in the order above.
    

### Build the docs
1.  From the the parent `client-superstaq` directory. Run `./checks/build_docs.py`
    Note: You may encounter an error telling you to install pandoc. This means you need to do a systems-level install of pandoc. You can do so by following the directions [here](https://pandoc.org/installing.html). Once done, repeat steps 2-3.
0. `cd` into the `docs` folder.
0. Run `open build/html/index.html`
 
## How to update the docs
1. Make sure you are on the `main` branch in `client-superstaq`.
0. Create a new branch off of `main` in which to make your updates.
0. Make any relevant updates.
0. Push all commits and create a Pull Request.
0. Request the relevant people to review your Pull Request.
0. After your Pull Request has been reviewed, merge in your branch.
