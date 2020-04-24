#!/usr/bin/env python3
import os
import click
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import deepmp.utils as ut


def acc_test_single(data, labels, model_file):
    model = load_model(model_file)
    test_loss, test_acc = model.evaluate(data, tf.convert_to_tensor(labels))

    return test_acc


def acc_test_joint(data_seq, labels_seq, model_seq, 
    data_err, labels_err, model_err):

    assert labels_err.all() == labels_seq.all()
    labels = labels_seq

    model_seq = load_model(model_seq)
    model_err = load_model(model_err)

    seq_pred = model_seq.predict(data_seq).flatten()
    err_pred = model_err.predict(data_err).flatten()
    
    inferred = np.zeros(len(seq_pred))
    #TODO improve
    inferred[np.argwhere(seq_pred > 0.5)] = 1
    inferred[np.argwhere(err_pred > 0.5)] = 1

    test_acc = round(1 - np.argwhere(labels != inferred).shape[0] / len(labels), 5)

    return test_acc
    

def call_mods(model, test_file, model_err, model_seq, one_hot_embedding, 
    kmer_sequence, output):
    #TODO <JB, MC> improve way to combine both methods
    if model == 'seq':
        data_seq, labels_seq = ut.get_data_sequence(
            test_file, kmer_sequence, one_hot_embedding
        )
        test_acc = acc_test_single(data_seq, labels_seq, model_seq)

    elif model == 'err':
        data_err, labels_err = ut.load_error_data(test_file)
        test_acc = acc_test_single(data_err, labels_err, model_err)

    elif model == 'joint':
        import pdb;pdb.set_trace()
        data_seq, labels_seq = ut.get_data_sequence(
            test_file, kmer_sequence, one_hot_embedding
        )
        data_err, labels_err = ut.load_error_data(test_file)
        test_acc = acc_test_joint(data_seq, labels_seq, model_seq, data_err, 
            labels_err, model_err)

    ut._write_to_file(
        os.path.join(output, 'accuracy.txt'), test_acc
    )
