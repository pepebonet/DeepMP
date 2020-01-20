
from os import path
from setuptools import setup, find_packages


VERSION = "0.1"
DESCRIPTION = "Deeplearning tool"

directory = path.dirname(path.abspath(__file__))

# Get requirements from the requirements.txt file
with open(path.join(directory, 'requirements.txt')) as f:
    required = f.read().splitlines()


# Get the long description from the README file
with open(path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='DeepMP',
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="",
    author="Mandi Chen & Jose Bonet",
    author_email="-",
    packages=find_packages(),
    install_requires=required,
    entry_points={
        'console_scripts': [
            'DeepMP = deepmp.DeepMP:cli',
        ]
    }
)