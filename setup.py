'''Setup file for the Atlas Client'''
from setuptools import setup, find_packages
description = 'The offical Nomic python client.'
setup(
    name='nomic',
    version='1.0.22',
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
        'cohere'
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
            "twine"
        ],
    },
    entry_points={
        'console_scripts': ['nomic=nomic.cli:cli'],
    },
    include_package_data=True
)



