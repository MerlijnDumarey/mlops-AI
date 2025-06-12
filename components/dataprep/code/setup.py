from setuptools import setup

setup(
    name="mlops_cleaning",
    version="0.1.0",
    py_modules=["mlops_cleaning"],
    install_requires=[
        "h5py",
        "pandas", 
        "numpy",
        "scikit-learn"
    ],
    url="https://github.com/MerlijnDumarey/mlops-AI",
    author="Merlijn Dumarey",
    author_email="merlijn.dumarey@student.howest.be",
)