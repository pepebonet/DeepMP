#!/usr/bin/env python3
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from deepmp.utils import kmer2code
from deepmp.model import *

def preprocess(train_file):
    vocab_size = 1024
    embedding_size = 128

    df = pd.read_csv(train_file, delimiter = "\t",names = ['chrom','pos',
                                'strand','pos_in_strand','readname','read_strand',
                                'k_mer','signal_means','signal_stds','signal_lens',
                                'cent_signals','methy_label'])
    df = df.dropna()
    kmer = df['k_mer'].apply(kmer2code)
    base_mean = df['signal_means'].apply(lambda x : [pd.to_numeric(i,downcast= tf.float32) for i in x.split(",")])
    base_std = df['signal_stds'].apply(lambda x : [pd.to_numeric(i,downcast= tf.float32) for i in x.split(",")])
    base_signal_len = df['signal_lens'].apply(lambda x : [pd.to_numeric(i,downcast= tf.float32) for i in x.split(",")])
    #df['cent_signals'] = df['cent_signals'].apply(lambda x : [pd.to_numeric(i) for i in x.split(",")])
    label = df['methy_label']

    return np.stack(kmer), np.stack(base_mean), np.stack(base_std), np.stack(base_signal_len), label


def train(train_file):

    kmer = 17
    bases, signal_means, signal_stds, signal_lens, label = preprocess(train_file)
    model = get_lstm_model(kmer)
    model.fit([bases, signal_means, signal_stds, signal_lens], label, batch_size=120, epochs=100,
                            callbacks = [EarlyStopping(monitor='val_loss', patience=15)],
                            validation_split=0.2, shuffle=True)
    return None



train("../data/extraction_outputs/val.tsv")
