from setuptools import find_packages, setup

# read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="lib",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
)
