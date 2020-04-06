#!/usr/bin/env python3

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from deepmp.utils import kmer2code


def preprocess_sequence(csv_file):

    df = pd.read_csv(csv_file, delimiter = "\t",names = ['chrom','pos',
                                'strand','pos_in_strand','readname','read_strand',
                                'kmer','signal_means','signal_stds','signal_lens',
                                'cent_signals','methy_label'])
    df = df.dropna()
    kmer = df['kmer'].apply(kmer2code)
    base_mean = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_means'].values]
    base_std = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_stds'].values]
    base_signal_len = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_lens'].values]
    label = df['methy_label']

    file_name = csv_file.rsplit('.',1)[0]

    with h5py.File(file_name+'_seq.h5', 'a') as hf:
        hf.create_dataset("kmer",  data=np.stack(kmer))
        hf.create_dataset("signal_means",  data=np.stack(base_mean))
        hf.create_dataset("signal_stds",  data=np.stack(base_std))
        hf.create_dataset("signal_lens",  data=np.stack(base_signal_len))
        hf.create_dataset("label",  data=label)

    return None


def preprocess_error(csv_file, feat):

    df = pd.read_csv(csv_file)

    X = df[df.columns[:-1]].values
    Y = df[df.columns[-1]].values

    file_name = csv_file.rsplit('.',1)[0]

    with h5py.File(file_name+'_err.h5', 'a') as hf:
        hf.create_dataset("X", data=X.reshape(X.shape[0], feat, 1))
        hf.create_dataset("Y", data=Y)

    return None
