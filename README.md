# DeepMP
DeepMP is a convolutional neural network (CNN)-based model that takes information from Nanopore signals and basecalling errors to detect whether a read is methylated or not. The model introduces a threshold-free position modification calling model sensitive to sites methylated at low frequency across cells.

![alt text](https://github.com/pepebonet/DeepMP/tree/release/docs/images/Figure_1_DeepMP.png)

# Contents
- [Installation](#Installation)
- [Usage](#Usage)
- [Example data](#Example-data)         

# Installation
## Clone repository
First download BnpC from the github repository:

        git clone https://github.com/pepebonet/DeepMP.git

## Install dependencies
We highly recommend to use a virtual environment for the installation and employment of DeepMP:

`Option 1:`

        conda create --name deepmp_2021 python>=3.6
        conda activate deepmp_2021
        pip install -e .

# Usage

### Feature extraction:
Features for the model need to be extracted. We present 3 different options: 

`Option 1:` Extract combined features
```
    DeepMP combined-extraction -fr path/to/fast5/files/ -re path/to/error/folder/ -rp path/to/reference/file/ -dn path/to/dict_read_names -m CG -o CpG_methylated_combined.tsv -ml 1 -cpu 56
```

`Option 2:` Extract sequence features
```
    DeepMP sequence-feature-extraction path/to/fast5/files/ -rp path/to/reference/file/ -m CG -o CpG_methylated.tsv -ml 1 -cpu 56
```

`Option 3:` Extract error features
```
    DeepMP single-read-error-extraction -ef path/to/error/folder/ -m CG -o output/error_features/ -l 1 -cpu 56
```

### Preprocess:
Extracted features are processed to get the information into h5 format which is the desired input for training, validation and testing. 

        DeepMP preprocess -f path/to/features.tsv  -ft combined -o output/folder/ -cpu 56


### Train models
Preprocessing is needed before training. Use `--model_type` flag to specify model for data preparation, choose between `seq` and `err`.
```
    DeepMP preprocess path/to/csv_file --model_type seq
```
Train sequence model from binary files.
```
    DeepMP train-nns -m seq -tf path/to/train/data -vf path/to/validation/data -md save/model/to/directory -ld save/log/to/directory
```
Train errors model from binary files.
```
    DeepMP train-nns -m err -tf path/to/train/data -vf path/to/validation/data -md save/model/to/directory -ld save/log/to/directory
```
Train joint model from binary files.
```
    DeepMP train-nns -m joint -tf path/to/train/data -vf path/to/validation/data -md save/model/to/directory -ld save/log/to/directory
```
- Use `-cp` to specify the checkpoint file while training model from checkpoints.


### Call modifications

Finally modifications for a given test set are obtained: 

```
    DeepMP call-modifications -m joint -tf path/to/test/data -md model/directory -o output/ -pos
```

- Specify model type with flag `-m`, choose from `seq, err, joint`(required).
- Add `-ef` for sequence model with both seq and error features.
- Add  `-pos` for test on positions.


# Example data
Step by step process to detect modifications employing DeepMP. 