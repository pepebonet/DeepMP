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

Training data (10M subset)

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/ecoli/extract_outputs/positions/10m_train.tsv`
`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/ecoli/extract_outputs/positions/10m_train_seq.h5`

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

#### 17-mer sequence feature

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
##### DeepMP on 10M ecoli
17-mer:

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/deepmp/models/ecoli/latest_17mer_seq/sequence_model/`

| Accuracy  |  Precision | Recall  |  F-score |
|---|---|---|---|
| 0.9340299963951111  |  0.9243327612129679 | 0.8985630651780053  |  0.9112657642146945 |


17-mer without number of signal:

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/deepmp/models/ecoli/seq_model_2_features/`

| Accuracy  |  Precision | Recall  |  F-score |
|---|---|---|---|
| 0.930446982383728  | 0.9116686886216159  |  0.9030062363554855 | 0.9073167871383186  |

##### Deepsignal on 10M ecoli
17-mer:

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/deepsignal/trained_model/ecoli_sites/`
``
``
``
