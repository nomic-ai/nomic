'''Setup file for the Atlas Client'''
import os
import sys
from setuptools import setup, find_packages
description = 'The official Nomic python client.'
    
setup(
    name='nomic',
    version='3.0.22',
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
        'pyjwt',
    ],
    extras_require={
        "local": [
            "gpt4all>=2.3.3,<3",
        ],
        'dev': [
            "nomic[local]",
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
        ],
        'aws': [
            'boto3',
            'sagemaker'
        ],
        'all': [
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
            'pyjwt',
            'boto3',
            'sagemaker'
        ]
    },
    entry_points={
        'console_scripts': ['nomic=nomic.cli:cli'],
    },
    include_package_data=True
)
