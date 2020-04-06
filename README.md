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
### Feature extraction:

```
    DeepMP context-extraction path/to/fast5/files/ -rp path/to/reference/file/ -m CG -o extraction_outputs/
```

### Train models
Preprocessing is needed before training. Use `--model_type` flag to specify model for data preparation, choose between `seq` and `err`.
```
    DeepMP preprocess path/to/csv_file --model_type seq
```
Train sequence model from binary files.
```
    DeepMP train-nns -ts path/to/train/data -vs path/to/validation/data -rnn lstm -md save/model/to/directory -ld save/log/to/directory
```
Train errors model from binary files.
```
    DeepMP train-nns -te path/to/train/data -ve path/to/validation/data -md save/model/to/directory -ld save/log/to/directory
```
### Plotting Arguments
### Other Arguments

# Example data
