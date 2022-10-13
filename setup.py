#!/usr/bin/env python

from distutils.core import setup

setup(
    name="dgf",
    version="1.0",
    description="tools for the project",
    author="Jay Baptista",
    author_email="jay.baptista@yale.edu",
    packages=["dgf"],
    install_requires=[
        "numpy",
    ],
)
