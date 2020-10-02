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

sequence model with error features

| Accuracy  |  Precision | Recall  |  F-score |
|---|---|---|---|
|0.9933199882507324   |   0.9938937471971299   |   0.9927409872257373   |   0.9933170327625466

joint model

| Accuracy  |  Precision | Recall  |  F-score |
|---|---|---|---|
|0.9978880286216736   |   0.996842995921039   |    0.9989161474039531   |   0.9978784948900093

##### Deepsignal on 10M ecoli

17-mer:

`/cfs/klemming/nobackup/m/mandiche/DeepMP-master/deepsignal/trained_model/ecoli_sites/`

##### DeepMP on 10M ecoli positions

sequence model

| Accuracy  |  Precision | Recall  |  F-score |
|---|---|---|---|
|0.9298456907272339   |   0.9165033796989729  |    0.9444389741753569  |    0.9302614989111034 |

sequence model with error features

| Accuracy  |  Precision | Recall  |  F-score |
|---|---|---|---|
|0.9931148290634155   |   0.9942634530038996   |   0.9918235816906541  |    0.9930420186788094 |

joint model

| Accuracy  |  Precision | Recall  |  F-score |
|---|---|---|---|
|0.9937736392021179  |    0.9937514786845623   |   0.9936752322555482  |    0.9937133540074811 |
