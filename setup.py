'''Setup file for the Atlas Client'''
from setuptools import setup, find_packages
description = 'The offical Nomic python client.'
setup(
    name='nomic',
    version='0.0.1',
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
        'ruamel.yaml',
        "click",
        "jsonlines"
    ],
    extras_require={
        'dev': [
            'black',
            'coverage',
            "pylint",
            "pytest",
            "sphinx",
            "myst-parser",
            "sphinx_rtd_theme",
            "furo",
            "sphinx-copybutton",
            "twine"
        ],
    },
    include_package_data=True
)



