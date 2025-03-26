from setuptools import setup, find_packages

setup(
    name="exect",
    version="0.1.0",
    description="End-to-end extreme classification with transformers",
    author="Johannes Melsbach",
    author_email="dev@melsbach.org",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
