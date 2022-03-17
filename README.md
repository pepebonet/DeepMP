# DeepMP
DeepMP is a convolutional neural network (CNN)-based model that takes information from Nanopore signals and basecalling errors to detect whether a read is methylated or not.  DeepMP's architecture consists of two different modules. First, the sequence module involves 6 1D convolutional layers with 256 1x4 filters. On the other hand, the error module comprises 3 1D convolutional layers and 3 locally connected layers both with 128 1x3 filters. Outputs are finally concatenated and inputted into a fully connected layer with 512 units.
Furthermore, DeepMP introduces a threshold-free position modification calling model sensitive to sites methylated at low frequency across cells. These novelties allow DeepMP to accurately detect methylated sites at read and position levels. 

<img src="docs/images/Figure_1_DeepMP.png" alt="alt text" width=1000 height="whatever">


# Contents
- [Installation](#Installation)
- [Usage](#Usage)
- [Example data](#Example-data)         

# Installation
## Clone repository
First download DeepMP from the github repository:

        git clone https://github.com/pepebonet/DeepMP.git

## Install dependencies
We highly recommend to use a virtual environment for the installation and employment of DeepMP:

`Create environment and install DeepMP:`

        conda create --name deepmp_2021 python=3.8
        conda activate deepmp_2021
        pip install -e .

`Install additional dependencies:`

        pip install ont-tombo
        pip install biopython
        conda install -c bioconda samtools
        conda install -c bioconda minimap2
        conda install -c anaconda openjdk
        conda install -c anaconda bottleneck
        
# Usage

This section highlights the main functionalities of DeepMP and the commands to run them. For a detailed insight into the whole process of predicting modifications go to the [Example data](#Example-data) section. 

### Feature extraction:
Features for the model need to be extracted. We present 3 different options: 

`Option 1:` Extract combined features
```
DeepMP combine-extraction -fr path/to/fast5/files/ -re path/to/error/folder/ -rp path/to/reference/file/ -dn path/to/dict_read_names -m CG -o CpG_methylated_combined.tsv -ml 1 -cpu 56
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
```
DeepMP preprocess -f path/to/features.tsv  -ft combined -o output/folder/ -cpu 56
```

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

Finally modifications are called for a given test data. Call modifications can be employed to be run in parallel or in a single CPU. 

Running call-modifications in parallel (faster and memory efficient):

```
DeepMP call-modifications -m joint -tf test/folder/preprocess/output/ -md model/directory -o output/ -pos -cpu 56
```

Running call-modifications in single CPU (deprecated):
```
DeepMP call-modifications -m joint -tf path/to/test/data/test_file.h5 -md model/directory -o output/ -pos
```

- Specify model type with flag `-m`, choose from `seq, err, joint`(required).
- Add `-ef` for sequence model with both seq and error features.
- Add  `-pos` for test on positions.

### Fast call modifications from fast5 files (under test)

DeepMP can call modifications from re-squiggled fast5 files in one step:

```
DeepMP fast-call-joint -f path/to/fast5s/ -ref path/to/reference/genome -md path/to/trained_model -j path/to/sam2tsv.jar
```

Note that this function is currently under test, stepwise process is recommanded. Please see the following example for details.

# Example data

Step by step process to detect modifications employing DeepMP on a sample (10 reads) of the E. coli dataset. Copy paste of all the commnads starting from the DeepMP folder will output the results on the read and position predictions. E. coli reads are located in: 

```
    docs/reads/
``` 

## Data Preparation

First, extract the fastqs from the reads (output paths need to be generated and updated for the following commands): 

```
python deepmp/miscellaneous/parse_fast5.py docs/reads/treated/ -ff5 True -o docs/output_example/error_features/treated/ -cpu 56 -bg Basecall_1D_001

python deepmp/miscellaneous/parse_fast5.py docs/reads/untreated/ -ff5 True -o docs/output_example/error_features/untreated/ -cpu 56 -bg Basecall_1D_001
```

Next, for both fastqs extracted, the reads are mapped to the reference genome. For that to be done, first load the following packages in your environment if you do not have them already. 

```
conda install -c bioconda samtools
conda install -c bioconda minimap2
conda install -c anaconda openjdk
```

Then, map reads to the genome with the reference genome available in /docs/ref/ : 
```
cd docs/output_example/error_features/treated/

 minimap2 -ax map-ont ../../../ref/Escherichia_coli_str_k_12_substr_mg1655.ASM584v2.dna.toplevel.fa Basecall_1D_001_BaseCalled_template.fastq | samtools view -hSb | samtools sort -@ 56 -o sample.bam

samtools index sample.bam
```

Repeat for the untreated folder:

```
cd ../untreated/

 minimap2 -ax map-ont ../../../ref/Escherichia_coli_str_k_12_substr_mg1655.ASM584v2.dna.toplevel.fa Basecall_1D_001_BaseCalled_template.fastq | samtools view -hSb | samtools sort -@ 56 -o sample.bam

samtools index sample.bam
```

The following step is to call the variants in both treated and untreated folders: 

```
samtools view -h -F 3844 sample.bam |  java -jar ../../../jvarkit/sam2tsv.jar -r ../../../ref/Escherichia_coli_str_k_12_substr_mg1655.ASM584v2.dna.toplevel.fa > sample.tsv

cd ../treated/

samtools view -h -F 3844 sample.bam |  java -jar ../../../jvarkit/sam2tsv.jar -r ../../../ref/Escherichia_coli_str_k_12_substr_mg1655.ASM584v2.dna.toplevel.fa > sample.tsv
```

To allow downstream parallelisation of the feature extraction, the generated sample.tsv is split into the different reads in a tmp folder. 

```
mkdir tmp
awk 'NR==1{ h=$0 }NR>1{ print (!a[$2]++? h ORS $0 : $0) > "tmp/"$1".txt" }' sample.tsv
```

Then go back to the untreated and repeat: 

```
cd ../untreated/
mkdir tmp
awk 'NR==1{ h=$0 }NR>1{ print (!a[$2]++? h ORS $0 : $0) > "tmp/"$1".txt" }' sample.tsv
```

As in  some scenarios the readnames of the fastqs do not match the fast5 readnames, a dictionary to parse each read pair may be needed. To do so, we employ the sequencing summary output after running Guppy. That is the case of E. coli: 

```
pip install biopython

python ../../../../deepmp/miscellaneous/get_dict_guppy.py -ss ../../../../docs/reads/sequencing_summary_untreated.txt -o dict_reads.pkl

cd ../treated/

python ../../../../deepmp/miscellaneous/get_dict_guppy.py -ss ../../../../docs/reads/sequencing_summary_treated.txt -o dict_reads.pkl
```

## Running DeepMP 

First, go back to the DeepMP folder to run the commands: 

```
cd ../../../../
```

With the data ready, we can now extract the combined features from the reads:

```
DeepMP combine-extraction -fr docs/reads/treated/ -re docs/output_example/error_features/treated/tmp/ -rp  docs/ref/Escherichia_coli_str_k_12_substr_mg1655.ASM584v2.dna.toplevel.fa -ml 1 -cpu 56 -m CG -dn docs/output_example/error_features/treated/dict_reads.pkl -o docs/output_example/treated_features.tsv

DeepMP combine-extraction -fr docs/reads/untreated/ -re docs/output_example/error_features/untreated/tmp/ -rp  docs/ref/Escherichia_coli_str_k_12_substr_mg1655.ASM584v2.dna.toplevel.fa -ml 0 -cpu 56 -m CG -dn docs/output_example/error_features/untreated/dict_reads.pkl -o docs/output_example/untreated_features.tsv
```

Once the features are extracted one can concat the resulting features of the treated and untreated samples into a single file to then perform the preprocess step: 

```
cd docs/output_example/

cat untreated_features.tsv treated_features.tsv > features.tsv

DeepMP preprocess -f features.tsv -ft combined -o . -cpu 56
```

With the test folder available we can now use one of the trained models to get the predictions from the model at read and position level: 

```
DeepMP call-modifications -m joint -md ../../trained_models/K12ER2925_joint_202106/ -tf test/ -pos -cpu 56 
```

The resulting files of the analysis should be the read and position calling predictions from DeepMP: 

```
read_predictions_joint_DeepMP.tsv
position_calling_joint_DeepMP.tsv
```
