from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="mlops_cleaning",
    version="0.1.0",
    py_modules=["mlops_cleaning"],
    install_requires=requirements,
)