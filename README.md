# DeepMP
DeepMP is a deeplearning algorithm to detect DNA modifications of Nanopore sequenced samples.

# Contents
- [Installation](#Installation)
- [Usage](#Usage)
- [Example data](#Example-data)         

# Installation
## Clone repository
First download BnpC from the github repository:

        git clone add/final/github/path

## Install dependencies
We highly recommend to use a virtual environment for the installation and employment of DeepMP:

`Option 1:`

        conda create --name environment_name --file requirements.txt

`Option 2:`

        pip install -e .

# Usage
Feature extraction:

```
    DeepMP context-extraction path/to/fast5/files/ -rp path/to/reference/file/ -m CG -o extraction_outputs/
```

## Arguments
### Feature Extraction Arguments
### Training Arguments
### Plotting Arguments
### Other Arguments

# Example data
