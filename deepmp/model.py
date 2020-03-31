#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *

def get_lstm_model(base_num):

    vocab_size = 1024
    embedding_size = 128
    weight_table = tf.compat.v1.get_variable("embedding", shape=[vocab_size, embedding_size], dtype=tf.float32,
                                            initializer=tf.compat.v1.truncated_normal_initializer(stddev=np.sqrt(2. / vocab_size)))

    base_int = Input(shape=(base_num,))
    means = Input(shape=(base_num,))
    stds = Input(shape=(base_num,))
    sanums = Input(shape=(base_num,))

    #embedded_base = tf.nn.embedding_lookup(weight_table, base_int)

    #vector = tf.concat([embedded_base, tf.reshape(means, [-1, base_num, 1]),
    #                                    tf.reshape(stds, [-1, base_num, 1]),
    #                                    tf.reshape(sanums, [-1, base_num, 1])], axis=2)
    vector = tf.concat([tf.reshape(base_int, [-1, base_num, 1]), tf.reshape(means, [-1, base_num, 1]),
                                        tf.reshape(stds, [-1, base_num, 1]),
                                        tf.reshape(sanums, [-1, base_num, 1])], axis=2)

    x =  Bidirectional(RNN([LSTMCell(100, dropout=0.2),LSTMCell(100, dropout=0.2),LSTMCell(100, dropout=0.2)]))(vector)

    x = Dense(50, activation='relu', use_bias=False)(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid', use_bias=False)(x)
    model = Model([base_int, means, stds, sanums], outputs=out)

    model.compile(optimizer=tf.keras.optimizers.Adam(),loss='binary_crossentropy',metrics=['acc'])
    print(model.summary())

    return model
