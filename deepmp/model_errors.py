#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *

#TODO <JB> Improve structure and run training on tegner
def get_cnn_model():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(32, 3, input_shape=(20,1), activation='relu'))
    model.add(tf.keras.layers.LocallyConnected1D(64, 3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(3))
    model.add(tf.keras.layers.Conv1D(32, 3, activation='relu'))
    model.add(tf.keras.layers.LocallyConnected1D(32, 3, activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False))

    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['acc']
    )
    print(model.summary())

    return model


def get_fcn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(20,)),
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])

    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['acc']
        )
    print(model.summary())

    return model