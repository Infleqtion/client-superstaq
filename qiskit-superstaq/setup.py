import io

from setuptools import find_packages, setup

# This reads the __version__ variable from qiskit_superstaq/_version.py
__version__ = ""
exec(open("qiskit_superstaq/_version.py").read())

name = "qiskit-superstaq"

description = "The Qiskit module that provides tools and access to Superstaq"

# README file as long_description.
long_description = io.open("README.md", encoding="utf-8").read()

# Read in requirements
requirements = open("requirements.txt").readlines()
requirements = [r.strip() for r in requirements]

# Read in dev requirements, installed with 'pip install qiskit-superstaq[dev]'
dev_requirements = open("dev-requirements.txt").readlines()
dev_requirements = [r.strip() for r in dev_requirements]

# Read in example requirements, installed with 'pip install qiskit-superstaq[examples]
example_requirements = open("example-requirements.txt").readlines()
example_requirements = [r.strip() for r in example_requirements]

# Sanity check
assert __version__, "Version string cannot be empty"


qiskit_superstaq_packages = ["qiskit_superstaq"] + [
    "qiskit_superstaq." + package for package in find_packages(where="qiskit_superstaq")
]

setup(
    name=name,
    version=__version__,
    url="https://github.com/Infleqtion/client-superstaq",
    author="Superstaq development team",
    author_email="superstaq@infleqtion.com",
    python_requires=(">=3.7.0"),
    install_requires=requirements,
    extras_require={"dev": dev_requirements, "examples": example_requirements},
    license="Apache 2",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=qiskit_superstaq_packages,
    package_data={"qiskit_superstaq": ["py.typed"]},
)
