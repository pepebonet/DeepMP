#!/usr/bin/env python3

import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from deepmp.utils import kmer2code
from deepmp.model_errors import *

def preprocess(file):
    df = pd.read_csv(file)
    return df[df.columns[:-1]], df[df.columns[-1]]


def train(train_file, val_file, model):

    X_train, Y_train = preprocess(train_file)
    X_val, Y_val  = preprocess(val_file)

    if model == 'cnn':
        model = get_cnn_model()
    else: 
        model = get_fcn_model()

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    model.fit(X_train, Y_train, batch_size=512, epochs=10,
                            callbacks = [tensorboard_callback],
                            validation_data=(X_val, Y_val))
    return 


train("../data/extraction_outputs/train_errors.tsv", \
    "../data/extraction_outputs/val_errors.tsv", 'cnn')