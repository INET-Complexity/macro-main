# macro-main/setup.py

from os.path import abspath, dirname, join

from setuptools import find_packages, setup

# Define the directory containing this file
this_dir = abspath(dirname(__file__))

# Read the long description from README.md
with open(join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read the global requirements.txt
with open(join(this_dir, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()


setup(
    name="macromodel",
    version="1.0.0",
    description="Combined macro-data and macro-model packages for agent-based modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    packages=find_packages(include=["macro_data", "macro_data.*", "macromodel", "macromodel.*"]),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",  # Specify the Python versions you support
)
