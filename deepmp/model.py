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
                                        tf.reshape(sanums, [-1, base_num, 1])], 
                                        axis=2)

    x = Bidirectional(RNN([LSTMCell(100, dropout=0.2), \
        LSTMCell(100, dropout=0.2),LSTMCell(100, dropout=0.2)]))(vector)

    x = Dense(50, activation='relu', use_bias=False)(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid', use_bias=False)(x)
    model = Model([embedded_bases, means, stds, sanums], outputs=out)

    model.compile(optimizer=tf.keras.optimizers.Adam(), 
        loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())

    return model


def get_cnn_model():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(128, 3, input_shape=(20,1), activation='relu'))
    model.add(tf.keras.layers.LocallyConnected1D(256, 3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(3))
    model.add(tf.keras.layers.Conv1D(128, 3, activation='relu'))
    model.add(tf.keras.layers.LocallyConnected1D(256, 3, activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False))

    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['acc']
    )
    print(model.summary())
    return model


#TODO <PB, MC> Need to obtain a proper joint model 
def joint_model(event_output, signal_output, error_output): 
    joint_input = tf.concat(
        [event_output, signal_output, error_output], axis=1) 

    joint_input_shape = joint_input.get_shape().as_list()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(joint_input_shape)))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False))

    return model