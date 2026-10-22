"""Packaging-hulp: platformspecifieke wheels.

Het pakket bevat gecompileerde EPANET-MSX-bibliotheken (lib/*.so, lib/*.dll).
Zo'n wheel mag niet als ``py3-none-any`` getagd worden, want dan zou pip hem
op elk platform installeren. Deze klasse zet de tag op ``py3-none-<platform>``.
Alle overige metadata staat in pyproject.toml.
"""
from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel


class PlatformBdistWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat


setup(cmdclass={"bdist_wheel": PlatformBdistWheel})
