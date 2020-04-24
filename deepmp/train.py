#!/usr/bin/env python3

import time
import h5py
import datetime
import numpy as np
import tensorflow as tf

from deepmp.model import *


def load_data(file):

    with h5py.File(file, 'r') as hf:
        bases = hf['kmer'][:]
        signal_means = hf['signal_means'][:]
        signal_stds = hf['signal_stds'][:]
        signal_lens = hf['signal_lens'][:]
        label = hf['label'][:]

    return bases, signal_means, signal_stds, signal_lens, label


def load_error_data(file):

    with h5py.File(file, 'r') as hf:
        X = hf['err_X'][:]
        Y = hf['err_Y'][:]

    return X, Y


def train_sequence(train_file, val_file, log_dir, model_dir, batch_size,
                                kmer, epochs, one_hot = False, rnn = None):

    embedding_flag = ""

    ## preprocess data
    bases, signal_means, signal_stds, signal_lens, label = load_data(train_file)
    v1, v2, v3, v4, vy  = load_data(val_file)

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
    model.fit(input_train, label, batch_size=batch_size, epochs=epochs,
                                                callbacks = [tensorboard_callback],
                                                validation_data = (input_val, vy))
    model.save(model_dir + "sequence_model")

    return None


def train_errors(train_file, val_file, log_dir, model_dir, feat,
    epochs, batch_size):
    X_train, Y_train = load_error_data(train_file)
    X_val, Y_val  = load_error_data(val_file)

    model = get_cnn_model(feat)

    log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_errors")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                            callbacks = [tensorboard_callback],
                            validation_data=(X_val, Y_val))
    model.save(model_dir + "error_model")

    return None

def train_jm(train_file, val_file, log_dir, model_dir, feat, batch_size, kmer, epochs):

    embedding_flag = ""

    ## preprocess data
    bases, signal_means, signal_stds, signal_lens, label = load_data(train_file)
    v1, v2, v3, v4, vy  = load_data(val_file)

    ## embed bases

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
    X_train, Y_train = load_error_data(train_file)
    X_val, Y_val  = load_error_data(val_file)

    print(input_train.shape)
    print(X_train.shape)

    ## train model
    model = joint_model(kmer, embedding_size, feat)
    log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_jm")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                            log_dir = log_dir, histogram_freq=1)

    model.fit([input_train, X_train], label, batch_size=batch_size, epochs=epochs,
                                                callbacks = [tensorboard_callback],
                                                validation_data = ([input_val,X_val], vy))
    model.save(model_dir + "joint_model")

    return None
