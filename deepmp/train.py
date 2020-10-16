#!/usr/bin/env python3

import datetime
import deepmp.utils as ut
from deepmp.model import *

embedding_size = 5

def train_sequence(train_file, val_file, log_dir, model_dir, batch_size,
                                kmer, epochs, err_features = False, rnn = None):

    input_train, label = ut.get_data_sequence(train_file, kmer, err_features)
    input_val, vy = ut.get_data_sequence(val_file, kmer, err_features)

    if err_features:
        features = 9
    else:
        features = 5

    ## train model
    if rnn:
        model = get_brnn_model(kmer, embedding_size, features, rnn_cell = rnn)

        log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_seq_lstm")
    else:
        model = get_sequence_model(kmer, embedding_size, features)

        log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_seq_cnn")

    ## save checkpoints
    model_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_seq_model")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                            log_dir = log_dir, histogram_freq=1)

    callback_list = [
                        tensorboard_callback,
                        tf.keras.callbacks.ModelCheckpoint(filepath= model_dir,
                                                            monitor='val_accuracy',
                                                            mode='max',
                                                            save_best_only=True)
                        ]

    model.fit(input_train, label, batch_size=batch_size, epochs=epochs,
                                                callbacks = callback_list,
                                                validation_data = (input_val, vy))
    model.save(model_dir)

    return None

#TODO DELETE in future
def train_errors(train_file, val_file, log_dir, model_dir, feat,
    epochs, batch_size):
    X_train, Y_train= ut.load_error_data(train_file)
    X_val, Y_val = ut.load_error_data(val_file)

    model = get_error_model(feat)

    log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_errors")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                            callbacks = [tensorboard_callback],
                            validation_data=(X_val, Y_val))
    model.save(model_dir + "error_model")

    return None



def train_errors_kmer(train_file, val_file, log_dir, model_dir, feat,
    epochs, batch_size):
    X_train, Y_train, bases_train = ut.load_error_data_kmer(train_file)
    X_val, Y_val, bases_val = ut.load_error_data_kmer(val_file)

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)

    embedded_bases = tf.one_hot(bases_train, embedding_size)
    val_bases = tf.one_hot(bases_val, embedding_size)

    size_feat = int(X_train.shape[1] / 5)

    input_train = tf.concat([embedded_bases, tf.reshape(X_train, [-1, 5, size_feat])], axis=2)
    input_val = tf.concat([val_bases, tf.reshape(X_val, [-1, 5, size_feat])], axis=2)

    model = get_cnn_model_kmer(embedding_size + size_feat)

    log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_errors")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    model.fit(input_train, Y_train, batch_size=batch_size, epochs=epochs,
                            callbacks = [tensorboard_callback],
                            validation_data=(input_val, Y_val))
    model.save(model_dir + "error_model")

    return None


def train_single_error(train_file, val_file, log_dir, model_dir, kmer,
    epochs, batch_size):

    input_train, label = ut.get_data_errors(train_file, kmer)
    input_val, vy = ut.get_data_errors(val_file, kmer)

    model = get_single_err_model(kmer)

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
                                                            save_best_only=True)
                        ]

    model.fit(input_train, label, batch_size=batch_size, epochs=epochs,
                                                callbacks = callback_list,
                                                validation_data = (input_val, vy))
    model.save(model_dir)

    return None


def train_jm(train_file, val_file, log_dir, model_dir, batch_size, kmer, epochs):

    input_train_seq, input_train_err, label = ut.get_data_jm(train_file, kmer)
    input_val_seq, input_val_err, vy = ut.get_data_jm(val_file, kmer)

    ## train model
    model = joint_model(kmer, embedding_size)

    log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_jm")
    model_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S_jm_model")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                            log_dir = log_dir, histogram_freq=1)
    callback_list = [
                        tensorboard_callback,
                        tf.keras.callbacks.ModelCheckpoint(filepath= model_dir,
                                                            monitor='val_accuracy',
                                                            mode='max',
                                                            save_best_only=True)
                        ]

    model.fit([input_train_seq, input_train_err], label, batch_size=batch_size, epochs=epochs,
                                                callbacks = callback_list,
                                                validation_data = ([input_val_seq, input_val_err], vy))
    model.save(model_dir)

    return None
