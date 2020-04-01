#!/usr/bin/env python3
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.callbacks import EarlyStopping
from deepmp.utils import kmer2code
from deepmp.model import *

def preprocess(csv_file, vocab_size, embedding_size):

    weight_table = tf.compat.v1.get_variable("embedding", shape=[vocab_size, embedding_size], dtype=tf.float32,
                                            initializer=tf.compat.v1.truncated_normal_initializer(stddev=np.sqrt(2. / vocab_size)))

    df = pd.read_csv(csv_file, delimiter = "\t",names = ['chrom','pos',
                                'strand','pos_in_strand','readname','read_strand',
                                'k_mer','signal_means','signal_stds','signal_lens',
                                'cent_signals','methy_label'])
    df = df.dropna()
    kmer = df['k_mer'].apply(kmer2code)
    embedded_bases = tf.nn.embedding_lookup(weight_table, tf.dtypes.cast(np.stack(kmer), tf.int32))
    base_mean = df['signal_means'].apply(lambda x : [pd.to_numeric(i,downcast= tf.float32) for i in x.split(",")])
    base_std = df['signal_stds'].apply(lambda x : [pd.to_numeric(i,downcast= tf.float32) for i in x.split(",")])
    base_signal_len = df['signal_lens'].apply(lambda x : [pd.to_numeric(i,downcast= tf.float32) for i in x.split(",")])
    #df['cent_signals'] = df['cent_signals'].apply(lambda x : [pd.to_numeric(i) for i in x.split(",")])
    label = df['methy_label']

    return embedded_bases, np.stack(base_mean), np.stack(base_std), np.stack(base_signal_len), label


def train(train_file, val_file):

    kmer = 17
    vocab_size = 1024
    embedding_size = 128

    bases, signal_means, signal_stds, signal_lens, label = preprocess(train_file, vocab_size, embedding_size)
    v1, v2, v3, v4, vy  = preprocess(val_file, vocab_size, embedding_size)
    model = get_lstm_model(kmer, embedding_size)

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit([bases, signal_means, signal_stds, signal_lens], label, batch_size=512, epochs=10,
                            callbacks = [tensorboard_callback],
                            validation_data=([v1, v2, v3, v4], vy))
    return None



train("../data/extraction_outputs/train.tsv","../data/extraction_outputs/val.tsv")
