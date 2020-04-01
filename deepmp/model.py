#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *

def get_lstm_model(base_num, embedding_size):


    embedded_bases = Input(shape=(base_num,embedding_size))
    means = Input(shape=(base_num,))
    stds = Input(shape=(base_num,))
    sanums = Input(shape=(base_num,))

    vector = tf.concat([embedded_bases, tf.reshape(means, [-1, base_num, 1]),
                                        tf.reshape(stds, [-1, base_num, 1]),
                                        tf.reshape(sanums, [-1, base_num, 1])], axis=2)

    x =  Bidirectional(RNN([LSTMCell(100, dropout=0.2),LSTMCell(100, dropout=0.2),LSTMCell(100, dropout=0.2)]))(vector)

    x = Dense(50, activation='relu', use_bias=False)(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid', use_bias=False)(x)
    model = Model([embedded_bases, means, stds, sanums], outputs=out)

    model.compile(optimizer=tf.keras.optimizers.Adam(),loss='binary_crossentropy',metrics=['acc'])
    print(model.summary())

    return model
