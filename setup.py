import io
from setuptools import find_packages, setup
import os

# This reads the __version__ variable from supermarq/_version.py
__version__ = ""
exec(open("supermarq/_version.py").read())

name = "SupermarQ"

description = "SupermarQ is a scalable, application-centric quantum benchmarking suite."

# README file as long_description.
long_description = io.open("README.md", encoding="utf-8").read()


# Read in requirements
requirements = open(os.path.dirname(os.path.realpath(__file__)) + "/requirements.txt").readlines()
requirements = [r.strip() for r in requirements]
dev_requirements = open(
    os.path.dirname(os.path.realpath(__file__)) + "/dev-requirements.txt"
).readlines()
dev_requirements = [r.strip() for r in dev_requirements]

supermarq_packages = ["supermarq"] + [
    "supermarq." + package for package in find_packages(where="supermarq")
]

# Sanity check
assert __version__, "Version string cannot be empty"

setup(
    name=name,
    version=__version__,
    url="https://github.com/SupertechLabs/SupermarQ",
    author="Super.tech",
    author_email="pranav@super.tech",
    python_requires=(">=3.8.0"),
    install_requires=requirements,
    extras_require={
        "dev_env": dev_requirements,
    },
    license="Apache 2",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=supermarq_packages,
    package_data={"supermarq": ["py.typed"]},
)
