#!/usr/bin/env python3
import os
import h5py
import click
import numpy as np
import pandas as pd
import tensorflow as tf
from deepmp.utils import kmer2code
from sklearn.model_selection import train_test_split


def get_data(treated, untreated, names=''):
    if names:
        treat = pd.read_csv(treated, delimiter = "\t", names=names)
        untreat = pd.read_csv(untreated, delimiter = "\t", names=names)
    else:
        treat = pd.read_csv(treated, sep=',')
        untreat = pd.read_csv(untreated, sep=',')
    return treat, untreat


def get_merge_data(errors, sequence):
    errors['pos'] = errors['pos'].astype('int64')
    return pd.merge(sequence, errors, on='pos', how='inner')
    

def get_training_test_data(df):
    train, test = train_test_split(df, test_size=0.05, random_state=0)
    train, val = train_test_split(train, test_size=test.shape[0], random_state=0)
    
    train_seq = train[train.columns[:12]]; train_err = train[train.columns[12:]]
    test_seq = test[test.columns[:12]]; test_err = test[test.columns[12:]]
    val_seq = val[val.columns[:12]]; val_err = val[val.columns[12:]]

    return [(train_seq, train_err, 'train'), (test_seq, test_err, 'test'), 
        (val_seq, val_err, 'val')]


def preprocess_sequence(df, output, file):

    df = df.dropna()
    kmer = df['kmer'].apply(kmer2code)
    base_mean = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_means'].values]
    base_std = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_stds'].values]
    base_signal_len = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_lens'].values]
    label = df['methyl_label']

    file_name = os.path.join(output, '{}_seq.h5'.format(file))

    with h5py.File(file_name, 'a') as hf:
        hf.create_dataset("kmer",  data=np.stack(kmer))
        hf.create_dataset("signal_means",  data=np.stack(base_mean))
        hf.create_dataset("signal_stds",  data=np.stack(base_std))
        hf.create_dataset("signal_lens",  data=np.stack(base_signal_len))
        hf.create_dataset("label",  data=label)

    return None


def preprocess_error(df, feat, output, file):

    X = df[df.columns[:-1]].values
    Y = df[df.columns[-1]].values

    file_name = os.path.join(output, '{}_err.h5'.format(file))

    with h5py.File(file_name, 'a') as hf:
        hf.create_dataset("X", data=X.reshape(X.shape[0], feat, 1))
        hf.create_dataset("Y", data=Y)

    return None


# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='Merge error and sequence features')
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
def main(error_treated, error_untreated, sequence_treated, 
    sequence_untreated, num_err_feat, output):
    #TODO add conditions if there is only seq or err and not both
    seq_treat, seq_untreat = get_data(sequence_treated, sequence_untreated, 
        names=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 
        'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_lens', 
        'cent_signals', 'methyl_label'])
    err_treat, err_untreat = get_data(error_treated, error_untreated)
    
    treat_merge = get_merge_data(err_treat, seq_treat)
    untreat_merge = get_merge_data(err_untreat, seq_untreat)
    
    data = get_training_test_data(pd.concat([treat_merge, untreat_merge]))

    for el in data:
        preprocess_sequence(el[0], output, el[2])
        preprocess_error(el[1], num_err_feat, output, el[2])

    
if __name__ == "__main__":
    main()