import os
import glob
import setuptools
from distutils.core import setup

with open("README.md", 'r') as readme:
    long_description = readme.read()

# to include data in the package, use MANIFEST.in

setup(
    name='vivarium-chemotaxis',
    version='0.0.1',
    packages=[
        'chemotaxis',
        'chemotaxis.composites',
        'chemotaxis.experiments',
        'chemotaxis.processes'],
    author='Eran Agmon, Ryan Spangler',
    author_email='eagmon@stanford.edu, ryan.spangler@gmail.com',
    url='https://github.com/vivarium-collective/vivarium-chemotaxis',
    license='MIT',
    entry_points={
        'console_scripts': []},
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'vivarium-core==0.0.29',
        'vivarium-cell==0.0.23'])
