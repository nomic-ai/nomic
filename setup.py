'''Setup file for the Atlas Client'''
import os
import sys
from setuptools import setup, find_packages
description = 'The official Nomic python client.'
    
setup(
    name='nomic',
    version='3.0.8',
    url='https://github.com/nomic-ai/nomic',
    description=description,
    long_description=description,
    packages=find_packages(),
    author_email="support@nomic.ai",
    author="nomic.ai",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    install_requires=[
        "click",
        "jsonlines",
        "loguru",
        'rich',
        'requests',
        'numpy',
        'pandas',
        'pydantic',
        'tqdm',
        'pyarrow',
        'pillow',
        'pyjwt'
    ],
    extras_require={
        'dev': [
            'black',
            'coverage',
            "pylint",
            "pytest",
            "myst-parser",
            "mkdocs-material",
            "mkautodoc",
            "twine",
            "mkdocstrings[python]",
            "mkdocs-jupyter",
            "pillow",
            "cairosvg"
        ]

    },
    entry_points={
        'console_scripts': ['nomic=nomic.cli:cli'],
    },
    include_package_data=True
)
