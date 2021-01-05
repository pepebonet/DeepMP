# Data
## E.Coli Data

#### Fast5

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/ecoli/training/ecoli_er2925.pcr.r9.timp.061716.fast5`
`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/ecoli/training/ecoli_er2925.pcr_MSssI.r9.timp.061716.fast5`

#### 17-mer sequence feature

The whole set

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/ecoli/training/seq_feature/full.tsv`

Training data (full)

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/ecoli/extract_outputs/positions/train.tsv`
`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/ecoli/extract_outputs/positions/train_seq.h5`


Site-based validation data

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/ecoli/extract_outputs/positions/val.tsv`
`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/ecoli/extract_outputs/positions/val_seq.h5`

Site-based test data

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/ecoli/extract_outputs/positions/test.tsv`
`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/ecoli/extract_outputs/positions/test_seq.h5`

Read-based test data

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/ecoli/extract_outputs/positions/1m_test.tsv`
`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/ecoli/extract_outputs/positions/1m_test_seq.h5`

#### single read error feature

treated

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/ecoli/single_read_error/whole_set/single_error_features/treated/template/single_read_errors.tsv`

untreated

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/ecoli/single_read_error/whole_set/single_error_features/untreated/template/single_read_errors.tsv`

#### combined feature

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/ecoli/extract_outputs/combined_features/train_test_val_split/`


## Human data
### Simpson's

#### Fast5

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/homo_sapiens/training/NA12878.pcr.r9.timp.081016.fast5`
`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/homo_sapiens/training/NA12878.pcr_MSssI.r9.timp.081016.fast5`

#### sequence feature

The whole set

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/homo_sapiens/training/seq_feature/full.tsv`
`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/homo_sapiens/training/seq_feature/data_full_seq.h5`

Site-based data

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/homo_sapiens/extraction_outputs/position_based`

### PRJEB23027

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/PRJEB23027/`

## E.Coli + Human training data (DeepMod trained)

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/deepmod_training_set/train.tsv`
`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/deepmod_training_set/train_seq.h5`

# Trained models

#### On Norwich 10ep

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/deepmp/models/20210104-170947_jm_model`

| Accuracy           | Precision         | Recall             | F-score            |
|--------------------|-------------------|--------------------|--------------------|
| 0.9176812171936035 | 0.939413472673029 | 0.9239651244992538 | 0.9316252612880689 |

#### On UCSC 7ep

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/deepmp/models/20210105-035844_jm_model`

| Accuracy           | Precision          | Recall             | F-score            |
|--------------------|--------------------|--------------------|--------------------|
| 0.9225872755050659 | 0.9359483323073174 | 0.9374880161759835 | 0.9367175415469722 |

