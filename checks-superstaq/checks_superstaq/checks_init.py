import importlib.resources
import click
import pathlib

import site

import importlib
import shutil

from .configs import CONFIG_FILE, TEMPLATE_FILE

CHECKS_FILE = "_checks.py"


@click.command()
def checks_init() -> None:

    checks_folder = pathlib.Path("checks")
    if not checks_folder.exists():
        checks_folder.mkdir()

    site.addsitedir(checks_folder)

    with importlib.resources.path("checks_superstaq", CHECKS_FILE) as base_checks_file:
        shutil.copyfile(base_checks_file, checks_folder/"checks.py")

    with importlib.resources.path("checks_superstaq", TEMPLATE_FILE) as template_toml:
        if pathlib.Path(CONFIG_FILE).exists():
            with pathlib.Path(CHECKS_FILE).open("a") as dst_file:
                with template_toml.open("w") as src_file:
                    dst_file.write(src_file.read())
        else:
            shutil.copyfile(template_toml, CONFIG_FILE)


@click.command()
def pytest() -> None:
    import checks.checks
    checks.checks.pytest()
