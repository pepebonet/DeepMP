#!/usr/bin/env python3

import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from deepmp.utils import kmer2code
from deepmp.model import *

def preprocess(file, feat):
    df = pd.read_csv(file)

    X = df[df.columns[:-1]].values
    Y = df[df.columns[-1]].values

    return X.reshape(X.shape[0], feat, 1), Y


def train(train_file, val_file):

    X_train, Y_train = preprocess(train_file, feat=20)
    X_val, Y_val  = preprocess(val_file, feat=20)
    
    model = get_cnn_model()

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S_errors")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )
 
    model.fit(X_train, Y_train, batch_size=512, epochs=12,
                            callbacks = [tensorboard_callback],
                            validation_data=(X_val, Y_val))
    return 


train("/home/jbonet/Desktop/error_features/ecoli/train_errors.csv", \
    "/home/jbonet/Desktop/error_features/ecoli/val_errors.csv")