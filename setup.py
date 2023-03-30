'''Setup file for the Atlas Client'''
import os
from setuptools import setup, find_packages
description = 'The offical Nomic python client.'
setup(
    name='nomic',
    version='1.1.4',
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
        'pydantic',
        'wonderwords',
        'tqdm',
        'cohere',
        'pyarrow',
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
        ],
        'gpt4all': [
            'torch',
            'sentencepiece',
            f"transformers @ file://localhost/{os.getcwd()}/bin/transformers-4.28.0.dev0-py3-none-any.whl",
            f"peft @ file://localhost/{os.getcwd()}/bin/peft-0.3.0.dev0-py3-none-any.whl"
        ]
    },
    entry_points={
        'console_scripts': ['nomic=nomic.cli:cli'],
    },
    include_package_data=True
)
