#!/usr/bin/env python3
import os
import h5py
import click
import numpy as np
import pandas as pd
import tensorflow as tf
from deepmp.utils import kmer2code
from sklearn.model_selection import train_test_split


def get_data(treated, untreated, names='', nopos=False):
    if names:
        treat = pd.read_csv(treated, delimiter = "\t", names=names)
        untreat = pd.read_csv(untreated, delimiter = "\t", names=names)
    else:
        if nopos:
            treat = pd.read_csv(treated, sep=',').drop(columns=['pos'])
            untreat = pd.read_csv(untreated, sep=',').drop(columns=['pos'])
        else:
            treat = pd.read_csv(treated, sep=',')
            untreat = pd.read_csv(untreated, sep=',')
    return treat, untreat


def get_merge_data(errors, sequence):
    subset = pd.merge(sequence[0:50000], errors, on='pos', how='right')
    test = subset[subset['chrom'] == 'Chromosome'] 
    only_errors = subset[subset['chrom'] != 'Chromosome']
    only_errors = only_errors[only_errors.columns[12:]]

    return only_errors, test


def get_training_test_val(df):
    train, val = train_test_split(df, test_size=0.05, random_state=0)
    return [(train, 'train'), (val, 'val')]


def save_data_seq_tsv(data, output):
    out_file = os.path.join(output, 'test_for_deepsignal.tsv')
    data['methyl_label'] = data['methyl_label'].astype('int32')
    data.to_csv(out_file, sep='\t', header=None, index=None)


def preprocess_both(data, err_feat, output, file):

    data = data.dropna()
    data_seq = data[data.columns[:12]]
    data_err = data[data.columns[12:]]

    save_data_seq_tsv(data_seq, output)
    
    #sequence
    kmer = data_seq['kmer'].apply(kmer2code)
    base_mean = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in data_seq['signal_means'].values]
    base_std = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in data_seq['signal_stds'].values]
    base_signal_len = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in data_seq['signal_lens'].values]
    label = data_seq['methyl_label']

    #error
    X = data_err[data_err.columns[:-1]].values
    Y = data_err[data_err.columns[-1]].values

    file_name = os.path.join(output, '{}.h5'.format(file))

    with h5py.File(file_name, 'a') as hf:
        hf.create_dataset("kmer",  data=np.stack(kmer))
        hf.create_dataset("signal_means",  data=np.stack(base_mean))
        hf.create_dataset("signal_stds",  data=np.stack(base_std))
        hf.create_dataset("signal_lens",  data=np.stack(base_signal_len))
        hf.create_dataset("label",  data=label)
        hf.create_dataset("err_X", data=X.reshape(X.shape[0], err_feat, 1))
        hf.create_dataset("err_Y", data=Y)

    return None

def preprocess_error(df, feat, output, file):

    X = df[df.columns[:-1]].values
    Y = df[df.columns[-1]].values

    file_name = os.path.join(output, '{}_err.h5'.format(file))

    with h5py.File(file_name, 'a') as hf:
        hf.create_dataset("err_X", data=X.reshape(X.shape[0], feat, 1))
        hf.create_dataset("err_Y", data=Y)

    return None


@click.command(short_help='Merge features and preprocess data for NNs')
@click.option(
    '-ft', '--feature_type', required=True,
    type=click.Choice(['seq', 'err', 'both']),
    help='which features is the input corresponding to? To the sequence, '
    'to the errors or to both of them. If choice and files do not correlate '
    'errors will rise throughout the script'
)
@click.option(
    '-et', '--error-treated', default='', help='extracted error features'
)
@click.option(
    '-eu', '--error-untreated', default='', help='extracted error features'
)
@click.option(
    '-st', '--sequence-treated', default='', help='extracted sequence features'
)
@click.option(
    '-su', '--sequence-untreated', default='', help='extracted sequence features'
)
@click.option(
    '-nef', '--num-err-feat', default=20, help='# Error features to select'
)
@click.option(
    '-o', '--output', default='', help='Output file'
)
def main(feature_type, error_treated, error_untreated,
    sequence_treated, sequence_untreated, num_err_feat, output):

    seq_treat, seq_untreat = get_data(sequence_treated, sequence_untreated, 
            names=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_lens', 
            'cent_signals', 'methyl_label'])
    err_treat, err_untreat = get_data(error_treated, error_untreated)

    treat, test_treat = get_merge_data(err_treat, seq_treat)
    untreat, test_untreat = get_merge_data(err_untreat, seq_untreat)

    test_all = pd.concat([test_treat, test_untreat])
    test_all = test_all.sample(frac=1).reset_index(drop=True)
    data = get_training_test_val(pd.concat([treat, untreat]))
    data.append((test_all, 'test'))

    for el in data:
        if el[1] == 'test':
            preprocess_both(el[0], num_err_feat, output, el[1])
        else:
            preprocess_error(el[0], num_err_feat, output, el[1])

if __name__ == '__main__':
    main()