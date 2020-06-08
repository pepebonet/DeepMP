#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support

import deepmp.utils as ut
import deepmp.plots as pl


def acc_test_single(data, labels, model_file, score_av='binary'):
    model = load_model(model_file)
    test_loss, test_acc = model.evaluate(data, tf.convert_to_tensor(labels))

    pred =  model.predict(data).flatten()
    inferred = np.zeros(len(pred))
    inferred[np.argwhere(pred > 0.5)] = 1

    precision, recall, f_score, _ = precision_recall_fscore_support(
        labels, inferred, average=score_av
    )

    return [test_acc, precision, recall, f_score]


def get_accuracy_joint(inferred, err_pred, seq_pred, labels, score_av='binary'):
    probs = np.zeros(len(labels))
    for i in range(len(labels)):
        if err_pred[i] > 0.5 and seq_pred[i] > 0.5:
            inferred[i] = 1
            probs[i] = max(seq_pred[i], err_pred[i])
        elif err_pred[i] < 0.5 and seq_pred[i] < 0.5:
            inferred[i] = 0
            probs[i] = min(seq_pred[i], err_pred[i])
        else: 
            val = (err_pred[i] + seq_pred[i]) / 2
            if val > 0.5:
                inferred[i] = 1
                probs[i] = max(seq_pred[i], err_pred[i])
            else: 
                inferred[i] = 0
                probs[i] = min(seq_pred[i], err_pred[i])

    test_acc = round(1 - np.argwhere(labels != inferred).shape[0] / len(labels), 5)
    
    precision, recall, f_score, _ = precision_recall_fscore_support(
        labels, inferred, average=score_av
    )

    return [test_acc, precision, recall, f_score], probs


def acc_test_joint(data_seq, labels_seq, model_seq, 
    data_err, labels_err, model_err):

    assert labels_err.all() == labels_seq.all()
    labels = labels_seq

    model_seq = load_model(model_seq)
    model_err = load_model(model_err)

    seq_pred = model_seq.predict(data_seq).flatten()
    err_pred = model_err.predict(data_err).flatten()
    
    inferred = np.zeros(len(seq_pred))

    return get_accuracy_joint(inferred, err_pred, seq_pred, labels)


def save_output(acc, output):
    col_names = ['Accuracy', 'Precision', 'Recall', 'F-score']
    df = pd.DataFrame([acc], columns=col_names)
    df.to_csv(os.path.join(output, 'accuracy_measurements.txt'), index=False, sep='\t')


def save_probs(probs, labels, output):
    out_probs = os.path.join(output, 'test_pred_prob.txt')
    probs_to_save = pd.DataFrame(columns=['labels', 'probs'])
    probs_to_save['labels'] = labels
    probs_to_save['probs'] = probs
    probs_to_save.to_csv(out_probs, sep='\t', index=None)


def call_mods(model, test_file, model_err, model_seq, one_hot_embedding, 
    kmer_sequence, output, figures=False):

    if model == 'seq':
        data_seq, labels = ut.get_data_sequence(
            test_file, kmer_sequence, one_hot_embedding
        )
        acc = acc_test_single(data_seq, labels, model_seq)

    elif model == 'err':
        data_err, labels = ut.load_error_data(test_file)
        acc = acc_test_single(data_err, labels, model_err)

    elif model == 'joint':
        data_seq, labels_seq = ut.get_data_sequence(
            test_file, kmer_sequence, one_hot_embedding
        )
        data_err, labels_err = ut.load_error_data(test_file)
 
        acc, probs = acc_test_joint(data_seq, labels_seq, model_seq, data_err, 
            labels_err, model_err)
        
        labels = labels_seq
        save_probs(probs, labels, output)
        
    save_output(acc, output)
    
    
    if figures:
        out_fig = os.path.join(output, 'ROC_curve.png')
        pl.plot_ROC(labels, probs, out_fig)

