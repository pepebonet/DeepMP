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
    # import pdb;pdb.set_trace()
    try: 
        return pd.merge(sequence, errors, on='pos', how='inner')
    except:
        errors['pos'] = errors['pos'].astype('int64')
        return pd.merge(sequence, errors, on='pos', how='inner')
    

def get_training_test_val(df):
    train, test = train_test_split(df, test_size=0.05, random_state=0)
    train, val = train_test_split(train, test_size=test.shape[0], random_state=0)
    return [(train, 'train'), (test, 'test'), (val, 'val')]


def preprocess_sequence(df, output, file):

    df = df.dropna()
    kmer = df['kmer'].apply(kmer2code)
    base_mean = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_means'].values]
    base_std = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_stds'].values]
    base_median = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_median'].values]
    base_skew = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_skew'].values]
    base_kurt = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_kurt'].values]
    base_diff = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_diff'].values]
    base_signal_len = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_lens'].values]
    label = df['methyl_label']

    file_name = os.path.join(output, '{}_seq.h5'.format(file))

    with h5py.File(file_name, 'a') as hf:
        hf.create_dataset("kmer",  data=np.stack(kmer))
        hf.create_dataset("signal_means",  data=np.stack(base_mean))
        hf.create_dataset("signal_stds",  data=np.stack(base_std))
        hf.create_dataset("signal_median",  data=np.stack(base_median))
        hf.create_dataset("signal_skew",  data=np.stack(base_skew))
        hf.create_dataset("signal_kurt",  data=np.stack(base_kurt))
        hf.create_dataset("signal_diff",  data=np.stack(base_diff))
        hf.create_dataset("signal_lens",  data=np.stack(base_signal_len))
        hf.create_dataset("label",  data=label)

    return None


def preprocess_error(df, feat, output, file):

    X = df[df.columns[:-1]].values
    Y = df[df.columns[-1]].values

    file_name = os.path.join(output, '{}_err.h5'.format(file))

    with h5py.File(file_name, 'a') as hf:
        hf.create_dataset("err_X", data=X.reshape(X.shape[0], feat, 1))
        hf.create_dataset("err_Y", data=Y)

    return None


def preprocess_both(data, err_feat, output, file):

    data = data.dropna()
    data_seq = data[data.columns[:12]]
    data_err = data[data.columns[12:]]

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


def do_seq_err_preprocess(sequence_treated, sequence_untreated, 
    error_treated, error_untreated, output, num_err_feat):
    seq_treat, seq_untreat = get_data(sequence_treated, sequence_untreated, 
            names=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_lens', 
            'cent_signals', 'methyl_label'])
    err_treat, err_untreat = get_data(error_treated, error_untreated)
    
    treat = get_merge_data(err_treat, seq_treat)
    untreat = get_merge_data(err_untreat, seq_untreat)

    data = get_training_test_val(pd.concat([treat, untreat]))
    
    for el in data:
        preprocess_both(el[0], num_err_feat, output, el[1])


def do_single_preprocess(feature_type, sequence_treated, sequence_untreated, 
    error_treated, error_untreated, output, num_err_feat):
    if feature_type == 'err':
        treat, untreat = get_data(error_treated, error_untreated, nopos=True)
        data = get_training_test_val(pd.concat([treat, untreat]))
        for el in data:
            preprocess_error(el[0], num_err_feat, output, el[1])

    else: 
        treat, untreat = get_data(sequence_treated, sequence_untreated, 
            names=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_skew', 'signal_kurt', 'signal_diff',  
            'signal_lens', 'methyl_label', 'flag'])
        data = get_training_test_val(pd.concat([treat, untreat]))
        for el in data:
            preprocess_sequence(el[0], output, el[1])


def preprocess_combined(df, output, file):
    kmer = df['kmer'].apply(kmer2code)
    base_mean = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_means'].values]
    base_std = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_stds'].values]
    base_median = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_median'].values]
    base_skew = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_skew'].values]
    base_kurt = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_kurt'].values]
    base_diff = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_diff'].values]
    base_signal_len = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_lens'].values]
    cent_signals = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['cent_signals'].values]
    base_qual = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['qual'].values]
    base_mis = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['mis'].values]
    base_ins = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['ins'].values]
    base_del = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['del'].values]
    label = df['methyl_label']

    file_name = os.path.join(output, '{}_combined.h5'.format(file))

    with h5py.File(file_name, 'a') as hf:
        hf.create_dataset("kmer",  data=np.stack(kmer))
        hf.create_dataset("signal_means",  data=np.stack(base_mean))
        hf.create_dataset("signal_stds",  data=np.stack(base_std))
        hf.create_dataset("signal_median",  data=np.stack(base_median))
        hf.create_dataset("signal_skew",  data=np.stack(base_skew))
        hf.create_dataset("signal_kurt",  data=np.stack(base_kurt))
        hf.create_dataset("signal_diff",  data=np.stack(base_diff))
        hf.create_dataset("signal_lens",  data=np.stack(base_signal_len))
        hf.create_dataset("signal_central",  data=np.stack(cent_signals))
        hf.create_dataset('qual',  data=np.stack(base_qual))
        hf.create_dataset('mis',  data=np.stack(base_mis))
        hf.create_dataset('ins',  data=np.stack(base_ins))
        hf.create_dataset('del',  data=np.stack(base_del))
        hf.create_dataset('methyl_label',  data=label)

    return None


def save_tsv(df, output, file):
    file_name = os.path.join(output, '{}.tsv'.format(file))
    df.to_csv(file_name, sep='\t', index=None)


def do_combined_preprocess(treated, untreated, output, tsv_flag):

    treat, untreat = get_data(treated, untreated,
        names=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_skew', 'signal_kurt', 'signal_diff', 'signal_lens', 
            'cent_signals', 'qual', 'mis', 'ins', 'del', 'methyl_label', 'flag'])
    
    data = get_training_test_val(pd.concat([treat, untreat]))

    if tsv_flag:
        for el in data:
            save_tsv(el[0], output, el[1])
    for el in data:
        preprocess_combined(el[0], output, el[1])
        