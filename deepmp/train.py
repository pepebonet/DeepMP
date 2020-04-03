#!/usr/bin/env python3

import time
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
from deepmp.utils import kmer2code
from deepmp.model import *

def preprocess(csv_file):

    df = pd.read_csv(csv_file, delimiter = "\t",names = ['chrom','pos',
                                'strand','pos_in_strand','readname','read_strand',
                                'k_mer','signal_means','signal_stds','signal_lens',
                                'cent_signals','methy_label'])
    df = df.dropna()
    kmer = df['k_mer'].apply(kmer2code)
    base_mean = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_means'].values]
    base_std = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_stds'].values]
    base_signal_len = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_lens'].values]
    label = df['methy_label']

    return np.stack(kmer), np.stack(base_mean), np.stack(base_std), \
            np.stack(base_signal_len), label


def train_sequence(train_file, val_file, log_dir, model_dir,
                                            one_hot = False, rnn = None):

    kmer = 17
    embedding_flag = ""

    ## preprocess data
    bases, signal_means, signal_stds, signal_lens, label = preprocess(train_file)
    v1, v2, v3, v4, vy  = preprocess(val_file)

    ## embed bases
    if one_hot:
        embedding_size = 5
        embedding_flag += "_one-hot_embedded"
        embedded_bases = tf.one_hot(bases, embedding_size)
        val_bases = tf.one_hot(v1, embedding_size)

    else:
        vocab_size = 1024
        embedding_size = 128
        weight_table = tf.compat.v1.get_variable(
                                "embedding",
                                shape = [vocab_size, embedding_size],
                                dtype=tf.float32,
                                initializer = tf.compat.v1.truncated_normal_initializer(
                                stddev = np.sqrt(2. / vocab_size)
                                ))
        embedded_bases = tf.nn.embedding_lookup(weight_table, bases)
        val_bases = tf.nn.embedding_lookup(weight_table, v1)

    ## prepare inputs for NNs
    input_train = tf.concat([embedded_bases,
                                    tf.reshape(signal_means, [-1, kmer, 1]),
                                    tf.reshape(signal_stds, [-1, kmer, 1]),
                                    tf.reshape(signal_lens, [-1, kmer, 1])],
                                    axis=2)
    input_val = tf.concat([val_bases, tf.reshape(v2, [-1, kmer, 1]),
                                        tf.reshape(v3, [-1, kmer, 1]),
                                        tf.reshape(v4, [-1, kmer, 1])],
                                        axis=2)

    ## train model
    if rnn:
        model = get_brnn_model(kmer, embedding_size, rnn_cell = rnn)

        log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_lstm")\
                                                                + embedding_flag
    else:
        model = get_conv1d_model(kmer, embedding_size)

        log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_conv1d")\
                                                                + embedding_flag
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                            log_dir = log_dir, histogram_freq=1)
    model.fit(input_train, label, batch_size=512, epochs=10,
                                                callbacks = [tensorboard_callback],
                                                validation_data = (input_val, vy))
    model.save(model_dir + "sequence_model")

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

    model.fit(X_train, Y_train, batch_size=512, epochs=12,
                            callbacks = [tensorboard_callback],
                            validation_data=(X_val, Y_val))
    return None
