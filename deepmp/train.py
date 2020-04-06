#!/usr/bin/env python3
import os
import time
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.callbacks import EarlyStopping
from deepmp.utils import kmer2code
from deepmp.model import *

def preprocess(csv_file, vocab_size, embedding_size):

    weight_table = tf.compat.v1.get_variable(
        "embedding", shape=[vocab_size, embedding_size], dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=np.sqrt(2. / vocab_size)))

    df = pd.read_csv(csv_file, delimiter = "\t",names = ['chrom','pos',
                                'strand','pos_in_strand','readname','read_strand',
                                'k_mer','signal_means','signal_stds','signal_lens',
                                'cent_signals','methy_label'])
    df = df.dropna()

    kmer = df['k_mer'].apply(kmer2code)
    embedded_bases = tf.nn.embedding_lookup(
        weight_table, tf.dtypes.cast(np.stack(kmer), tf.int32)
    )

    base_mean = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_means'].values]
    base_std = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_stds'].values]
    base_signal_len = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_lens'].values]
    
    label = df['methy_label']

    return embedded_bases, np.stack(base_mean), np.stack(base_std), \
        np.stack(base_signal_len), label


def train_sequence(train_file, val_file):

    kmer = 17
    vocab_size = 1024
    embedding_size = 128

    bases, signal_means, signal_stds, signal_lens, label = preprocess(
        train_file, vocab_size, embedding_size)
    v1, v2, v3, v4, vy  = preprocess(val_file, vocab_size, embedding_size)
    model = get_lstm_model(kmer, embedding_size)

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S_lstm")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit([bases, signal_means, signal_stds, signal_lens], label, batch_size=512, epochs=10,
                            callbacks = [tensorboard_callback],
                            validation_data=([v1, v2, v3, v4], vy))
    return None


def preprocess_errors(file, feat):
    df = pd.read_csv(file)

    X = df[df.columns[:-1]].values
    Y = df[df.columns[-1]].values

    return X.reshape(X.shape[0], feat, 1), Y


def train_errors(train_file, val_file):

    X_train, Y_train = preprocess_errors(train_file, feat=20)
    X_val, Y_val  = preprocess_errors(val_file, feat=20)
    
    model = get_cnn_model()

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S_errors")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )
 
    model.fit(X_train, Y_train, batch_size=512, epochs=50,
                            callbacks = [tensorboard_callback],
                            validation_data=(X_val, Y_val))
    return 

