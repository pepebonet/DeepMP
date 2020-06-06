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
### Merge and preprocess:
Sequence and error features need to be combined so that the validation, test and train sets are obtained

        DeepMP merge-and-preprocess -et path/to/treated_error_features.csv -eu path/to/untreated_error_features.csv -st path/to/treated_sequence_features.tsv -su path/to/untreated_sequence_features.tsv -o output/ -ft both

### Feature extraction:

`Option 1:` Extract sequence features
```
    DeepMP sequence-feature-extraction path/to/fast5/files/ -rp path/to/reference/file/ -m CG -o extraction_outputs/ -ml 1
```

`Option 2:` Extract error features
```
    DeepMP error-extraction -ef path/to/errors/extracted/by/epinano -rp path/to/reference/file/ -m CG -o output/error_features/ -l 1
```

### Call modifications

```
    DeepMP call-modifications -model seq -tf path/to/test/data -one_hot -ms model/directory -o output/
```

### Train models
Preprocessing is needed before training. Use `--model_type` flag to specify model for data preparation, choose between `seq` and `err`.
```
    DeepMP preprocess path/to/csv_file --model_type seq
```
Train sequence model from binary files.
```
    DeepMP train-nns -model seq -tf path/to/train/data -vf path/to/validation/data -rnn lstm -md save/model/to/directory -ld save/log/to/directory
```
Train errors model from binary files.
```
    DeepMP train-nns -model err -tf path/to/train/data -vf path/to/validation/data -md save/model/to/directory -ld save/log/to/directory
```
Train joint model from binary files.
```
    DeepMP train-nns -model joint -tf path/to/train/data -vf path/to/validation/data -md save/model/to/directory -ld save/log/to/directory
```
### Plotting Arguments
### Other Arguments

# Example data
