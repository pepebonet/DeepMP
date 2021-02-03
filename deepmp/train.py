#!/usr/bin/env python3

import datetime
import deepmp.utils as ut
from deepmp.model import *

embedding_size = 5

def train_sequence(train_file, val_file, log_dir, model_dir, batch_size,
                                kmer, epochs, err_features = False, rnn = None, checkpoint_path = None):

    input_train, label = ut.get_data_sequence(train_file, kmer, err_features)
    input_val, vy = ut.get_data_sequence(val_file, kmer, err_features)

    if err_features:
        features = 8
    else:
        features = 4

    ## train model
    depth = embedding_size + features
    input_shape = (None, kmer, depth)

    if rnn:
        model = SequenceBRNN(256, 0.2, rnn_cell = rnn)
    else:
        model = SequenceCNN('conv', 6, 256, 4)

    model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=['accuracy'])
    model.build(input_shape)
    print(model.summary())
    if checkpoint_path:
        model.load_weights(checkpoint_path)

    log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_seq")

    ## save checkpoints
    model_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_seq_model")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                            log_dir = log_dir, histogram_freq=1)

    callback_list = [
                        tensorboard_callback,
                        tf.keras.callbacks.ModelCheckpoint(filepath= model_dir,
                                                            monitor='val_accuracy',
                                                            mode='max',
                                                            save_best_only=True,
                                                            save_weights_only= False)
                        ]

    model.fit(input_train, label, batch_size=batch_size, epochs=epochs,
                                                callbacks = callback_list,
                                                validation_data = (input_val, vy))
    model.save(model_dir)

    return None



def train_single_error(train_file, val_file, log_dir, model_dir, kmer,
    epochs, batch_size, checkpoint_path = None):

    input_train, label = ut.get_data_errors(train_file, kmer)
    input_val, vy = ut.get_data_errors(val_file, kmer)

    depth = 9
    input_shape = (None, kmer, depth)
    model = BCErrorCNN(3, 3, 128, 3)
    model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
    model.build(input_shape)
    print(model.summary())
    if checkpoint_path:
        model.load_weights(checkpoint_path)

    ## save checkpoints
    model_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_err_model")
    log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_err_read")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                                log_dir = log_dir, histogram_freq=1)
    callback_list = [
                        tensorboard_callback,
                        tf.keras.callbacks.ModelCheckpoint(filepath= model_dir,
                                                            monitor='val_accuracy',
                                                            mode='max',
                                                            save_best_only=True,
                                                            save_weights_only= False)
                        ]

    model.fit(input_train, label, batch_size=batch_size, epochs=epochs,
                                                callbacks = callback_list,
                                                validation_data = (input_val, vy))
    model.save(model_dir)

    return None


def train_jm(train_file, val_file, log_dir, model_dir, batch_size, kmer, epochs, checkpoint_path = None):

    input_train_seq, input_train_err, label = ut.get_data_jm(train_file, kmer)
    input_val_seq, input_val_err, vy = ut.get_data_jm(val_file, kmer)

    ## train model
    model = JointNN()
    model.compile(loss='binary_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.00125),
                   metrics=['accuracy'])
    input_shape = ([(None, kmer, 9), (None, kmer, 9)])
    model.build(input_shape)
    print(model.summary())
    if checkpoint_path:
        model.load_weights(checkpoint_path)

    log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_jm")
    model_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_jm_model")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                            log_dir = log_dir, histogram_freq=1)
    callback_list = [
                        tensorboard_callback,
                        tf.keras.callbacks.ModelCheckpoint(filepath= model_dir,
                                                            monitor='val_accuracy',
                                                            mode='max',
                                                            save_best_only=True,
                                                            save_weights_only= False)
                        ]

    model.fit([input_train_seq, input_train_err], label, batch_size=batch_size, epochs=epochs,
                                                callbacks = callback_list,
                                                validation_data = ([input_val_seq, input_val_err], vy))
    model.save(model_dir)

    return None
