# -*- coding: utf-8 -*-
from os.path import abspath, dirname, join

from setuptools import find_packages, setup

this_dir = abspath(dirname(__file__))

with open(join(this_dir, "README.md"), encoding="utf-8") as file:
    long_description = file.read()

with open(join(this_dir, "requirements.txt")) as f:
    requirements = f.read().split("\n")

setup(
    name="inet-macro-data",
    version="0.1.0",
    description="INET's MacroModel - Data",
    url="https://github.com/INET-Complexity/inet-macro-data",
    long_description_content_type="text/markdown",
    long_description=long_description,
    author="INET/Macrocosm",
    author_email="samuel.wiese@wolfson.ox.ac.uk",
    license="GPL 3 License",
    install_requires=requirements,
    packages=find_packages(exclude=["docs"]),
    include_package_data=True,
    package_data={"": ["*.yaml"]},
)
