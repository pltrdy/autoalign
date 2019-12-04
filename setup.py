#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='autoalign',
    description='Align then Summarize: Automatic Alignment Methods for Summarization Corpus Creation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.1.0rc1',
    packages=find_packages(),
    project_urls={
        "Documentation": "http://github.com/pltrdy/autoalign",
        "Source": "https://github.com/pltrdy/autoalign"
    },
    install_requires=[
        "torch",
        "stanfordnlp",
        "sklearn",
        "python-docx",
        "nltk",
        "rouge",
        "pathos",
        "public_meetings"
    ],
    package_data={
    },
)
