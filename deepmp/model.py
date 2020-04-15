#!/usr/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *

def get_brnn_model(base_num, embedding_size, rnn_cell = "lstm"):

    depth = embedding_size + 3
    input = Input(shape=(base_num, depth))

    if rnn_cell == "gru":
        x = Bidirectional(RNN([GRUCell(256, dropout=0.2), \
                GRUCell(256, dropout=0.2),GRUCell(256, dropout=0.2)]))(input)
    else:
        x = Bidirectional(RNN([LSTMCell(256, dropout=0.2), \
                LSTMCell(256, dropout=0.2),LSTMCell(256, dropout=0.2)]))(input)
    #x = Dense(50, activation='relu', use_bias=False)(x)
    x = Dense(512, activation='relu', use_bias=False)(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid', use_bias=False)(x)
    model = Model(input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())

    return model

def get_conv1d_model(base_num, embedding_size):

    depth = embedding_size + 3
    model = Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(base_num,depth)))
    model.add(tf.keras.layers.Conv1D(256, 5, activation='relu'))
    model.add(tf.keras.layers.LocallyConnected1D(128, 3, activation='relu'))
    model.add(tf.keras.layers.Conv1D(256, 3, activation='relu'))
    model.add(tf.keras.layers.LocallyConnected1D(128, 3, activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False))

    model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
    print(model.summary())
    return model


def get_cnn_model(feat):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(128, 3, input_shape=(feat, 1), activation='relu'))
    model.add(tf.keras.layers.LocallyConnected1D(256, 3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(2))
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
