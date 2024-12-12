"""Setup file for the Atlas Client"""

import os
import sys
from setuptools import setup, find_packages

description = "The official Nomic python client."

# Read README.md and remove tables and images
with open("README.md") as f:
    content = f.read()
    # Remove table sections including content
    while "<table>" in content and "</table>" in content:
        start = content.find("<table>")
        end = content.find("</table>") + 8
        content = content[:start] + content[end:]
    # Remove img tags and content
    while "<img" in content and ">" in content:
        start = content.find("<img")
        end = content.find(">", start) + 1
        content = content[:start] + content[end:]
    long_description = content

setup(
    name="nomic",
    version="3.3.4",
    url="https://github.com/nomic-ai/nomic",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["nomic", "nomic.*"]),
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
        "rich",
        "requests",
        "numpy",
        "pandas",
        "pydantic",
        "tqdm",
        "pyarrow",
        "pillow",
        "pyjwt",
    ],
    extras_require={
        "local": [
            "gpt4all>=2.5.0,<3",
        ],
        "aws": ["boto3", "sagemaker"],
        "all": [
            "nomic[local,aws]",
        ],
        "dev": [
            "nomic[all]",
            "black==24.3.0",
            "coverage",
            "pylint",
            "pytest",
            "isort",
            "pyright<=1.1.377",
            "myst-parser",
            "mkdocs-material",
            "mkautodoc",
            "twine",
            "mkdocstrings[python]",
            "mkdocs-jupyter",
            "pillow",
            "cairosvg",
            "pytorch-lightning",
            "pandas",
        ],
    },
    entry_points={
        "console_scripts": ["nomic=nomic.cli:cli"],
    },
    include_package_data=True,
)
