import io

from setuptools import find_packages, setup

# This reads the __version__ variable from general/_version.py
__version__ = ""
exec(open("general_superstaq/_version.py").read())

name = "general-superstaq"

description = "The general module that provides tools and access to SuperstaQ"

# README file as long_description.
long_description = io.open("README.md", encoding="utf-8").read()

# Read in requirements
requirements = open("requirements.txt").readlines()
requirements = [r.strip() for r in requirements]

# Read in dev requirements, installed with 'pip install general-superstaq[dev]'
dev_requirements = open("dev-requirements.txt").readlines()
dev_requirements = [r.strip() for r in dev_requirements]

# Sanity check
assert __version__, "Version string cannot be empty"


general_superstaq_packages = ["general_superstaq"] + [
    "general_superstaq." + package for package in find_packages(where="general_superstaq")
]

setup(
    name=name,
    version=__version__,
    url="https://github.com/SupertechLabs/general-superstaq",
    author="Super.tech",
    author_email="pranav@super.tech",
    python_requires=(">=3.6.0"),
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
    license="Apache 2",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=general_superstaq_packages,
    package_data={"general_superstaq": ["py.typed"]},
)
