#!/usr/bin/env python3

import time
import h5py
import datetime
import numpy as np
import tensorflow as tf

from deepmp.model import *


def load_seq_data(file):

    with h5py.File(file, 'r') as hf:
        bases = hf['kmer'][:]
        signal_means = hf['signal_means'][:]
        signal_stds = hf['signal_stds'][:]
        signal_medians = hf['signal_median']
        signal_range = hf['signal_diff']
        signal_lens = hf['signal_lens'][:]
        label = hf['label'][:]


    return bases, signal_means, signal_stds, signal_medians, signal_range, signal_lens, label


def load_error_data(file):

    with h5py.File(file, 'r') as hf:
        X = hf['err_X'][:]
        Y = hf['err_Y'][:]

    return X, Y

def load_err_read(file):

    with h5py.File(file, 'r') as hf:
        bases = hf['kmer'][:]
        base_qual = hf['qual'][:]
        base_mis = hf['mis'][:]
        base_ins = hf['ins'][:]
        base_del = hf['dele'][:]
        label = hf['methyl_label'][:]

    return bases, base_qual, base_mis, base_ins, base_del, label


def load_jm_data(file):

    with h5py.File(file, 'r') as hf:
        bases = hf['kmer'][:]
        signal_means = hf['signal_means'][:]
        signal_stds = hf['signal_stds'][:]
        signal_medians = hf['signal_median']
        signal_range = hf['signal_diff']
        signal_lens = hf['signal_lens'][:]
        base_qual = hf['qual'][:]
        base_mis = hf['mis'][:]
        base_ins = hf['ins'][:]
        base_del = hf['dele'][:]
        label = hf['methyl_label'][:]

    return bases, signal_means, signal_stds, signal_medians, signal_range,
            signal_lens, base_qual, base_mis, base_ins, base_del, label

def train_sequence(train_file, val_file, log_dir, model_dir, batch_size,
                                kmer, epochs, one_hot = False, rnn = None):

    embedding_flag = ""

    ## preprocess data
    bases, signal_means, signal_stds, signal_medians,
        signal_range, signal_lens, label = load_seq_data(train_file)
    v1, v2, v3, v4, v5, v6, vy  = load_seq_data(val_file)

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
                                    tf.reshape(signal_medians, [-1, kmer, 1]),
                                    tf.reshape(signal_range, [-1, kmer, 1])],
                                    tf.reshape(signal_lens, [-1, kmer, 1])],
                                    axis=2)
    input_val = tf.concat([val_bases,
                                    tf.reshape(v2, [-1, kmer, 1]),
                                    tf.reshape(v3, [-1, kmer, 1]),
                                    tf.reshape(v4, [-1, kmer, 1]),
                                    tf.reshape(v5, [-1, kmer, 1]),
                                    tf.reshape(v6, [-1, kmer, 1])],
                                        axis=2)

    ## train model
    if rnn:
        model = get_brnn_model(kmer, embedding_size, rnn_cell = rnn)

        log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_lstm")\
                                                                + embedding_flag
    else:
        model = get_sequence_model(kmer, embedding_size)

        log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_conv1d")\
                                                                + embedding_flag
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                            log_dir = log_dir, histogram_freq=1)
    model.fit(input_train, label, batch_size=batch_size, epochs=epochs,
                                                callbacks = [tensorboard_callback],
                                                validation_data = (input_val, vy))
    model.save(model_dir + "seq_model_5_features")

    return None


def train_errors(train_file, val_file, log_dir, model_dir, feat,
    epochs, batch_size):
    X_train, Y_train = load_error_data(train_file)
    X_val, Y_val  = load_error_data(val_file)

    model = get_error_model(feat)

    log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_errors")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                            callbacks = [tensorboard_callback],
                            validation_data=(X_val, Y_val))
    model.save(model_dir + "error_model")

    return None

def train_single_error(train_file, val_file, log_dir, model_dir, kmer,
    epochs, batch_size):

    bases, base_qual, base_mis, base_ins, base_del, label = load_err_read(train_file)
    v1, v2, v3, v4, v5, vy  = load_err_read(val_file)

    embedding_size = 5
    embedded_bases = tf.one_hot(bases, embedding_size)
    val_bases = tf.one_hot(v1, embedding_size)

    input_train = tf.concat([embedded_bases,tf.reshape(base_qual, [-1, kmer, 1]),
                                            tf.reshape(base_mis, [-1, kmer, 1]),
                                            tf.reshape(base_ins, [-1, kmer, 1]),
                                            tf.reshape(base_del, [-1, kmer, 1])],
                                            axis=2)

    input_val = tf.concat([val_bases, tf.reshape(v2, [-1, kmer, 1]),
                                            tf.reshape(v3, [-1, kmer, 1]),
                                            tf.reshape(v4, [-1, kmer, 1]),
                                            tf.reshape(v5, [-1, kmer, 1])],
                                            axis=2)

    model = get_single_err_model(kmer)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                                log_dir = log_dir, histogram_freq=1)

    model.fit(input_train, label, batch_size=batch_size, epochs=epochs,
                                                callbacks = [tensorboard_callback],
                                                validation_data = (input_val, vy))
    model.save(model_dir + "single_error_model")

    return None


def train_jm(train_file, val_file, log_dir, model_dir, batch_size, kmer, epochs):

    ## preprocess data
    bases, signal_means, signal_stds, signal_medians, signal_range,
            signal_lens, base_qual, base_mis, base_ins, base_del, label = load_jm_data(train_file)
    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, vy = load_jm_data(val_file)

    ## embed bases
    embedding_size = 5
    embedded_bases = tf.one_hot(bases, embedding_size)
    val_bases = tf.one_hot(v1, embedding_size)
    embedding_flag = ""

    ## prepare inputs for NNs
    input_train_seq = tf.concat([embedded_bases,
                                    tf.reshape(signal_means, [-1, kmer, 1]),
                                    tf.reshape(signal_stds, [-1, kmer, 1]),
                                    tf.reshape(signal_medians, [-1, kmer, 1]),
                                    tf.reshape(signal_range, [-1, kmer, 1])],
                                    tf.reshape(signal_lens, [-1, kmer, 1])],
                                    axis=2)
    input_val_seq = tf.concat([val_bases, tf.reshape(v2, [-1, kmer, 1]),
                                        tf.reshape(v3, [-1, kmer, 1]),
                                        tf.reshape(v4, [-1, kmer, 1]),
                                        tf.reshape(v5, [-1, kmer, 1]),
                                        tf.reshape(v6, [-1, kmer, 1])],
                                        axis=2)
    input_train_err = tf.concat([embedded_bases,tf.reshape(base_qual, [-1, kmer, 1]),
                                            tf.reshape(base_mis, [-1, kmer, 1]),
                                            tf.reshape(base_ins, [-1, kmer, 1]),
                                            tf.reshape(base_del, [-1, kmer, 1])],
                                            axis=2)
    input_val_err = tf.concat([val_bases, tf.reshape(v7, [-1, kmer, 1]),
                                            tf.reshape(v8, [-1, kmer, 1]),
                                            tf.reshape(v9, [-1, kmer, 1]),
                                            tf.reshape(v10, [-1, kmer, 1])],
                                            axis=2)

    ## train model
    model = joint_model(kmer, embedding_size)
    log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_jm")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                            log_dir = log_dir, histogram_freq=1)

    model.fit([input_train_seq, input_train_err], label, batch_size=batch_size, epochs=epochs,
                                                callbacks = [tensorboard_callback],
                                                validation_data = ([input_val_seq, input_val_err], vy))
    model.save(model_dir + "joint_model")

    return None
