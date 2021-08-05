import io

from setuptools import find_packages, setup

# This reads the __version__ variable from cirq/_version.py
__version__ = ""
exec(open("cirq_superstaq/_version.py").read())

name = "cirq-superstaq"

description = "The Cirq module that provides tools and access to SuperstaQ"

# README file as long_description.
long_description = io.open("README.md", encoding="utf-8").read()

# Read in requirements
requirements = open("requirements.txt").readlines()
requirements = [r.strip() for r in requirements]

# Sanity check
assert __version__, "Version string cannot be empty"


cirq_superstaq_packages = ["cirq_superstaq"] + [
    "cirq_superstaq." + package for package in find_packages(where="cirq_superstaq")
]


# Optional installations. For example, pip install cirq-superstaq[aqt] will include
# the package(s) needed for AQT use cases.
aqt_extra = ["qtrl"]
all_extra = aqt_extra

setup(
    name=name,
    version=__version__,
    url="https://github.com/SupertechLabs/cirq-superstaq",
    author="Super.tech",
    author_email="pranav@super.tech",
    python_requires=(">=3.8.0"),
    install_requires=requirements,
    license="N/A",
    description=description,
    long_description=long_description,
    packages=cirq_superstaq_packages,
    package_data={"cirq_superstaq": ["py.typed"]},
    extras_require={"aqt": aqt_extra, "all": all_extra},
)
