#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support

import deepmp.utils as ut
import deepmp.plots as pl
import deepmp.preprocess as pr


def acc_test_single(data, labels, model_file, score_av='binary'):
    model = load_model(model_file)
    test_loss, test_acc = model.evaluate(data, tf.convert_to_tensor(labels))

    pred =  model.predict(data).flatten()
    inferred = np.zeros(len(pred), dtype=int)
    inferred[np.argwhere(pred >= 0.5)] = 1

    precision, recall, f_score, _ = precision_recall_fscore_support(
        labels, inferred, average=score_av
    )
    
    return [test_acc, precision, recall, f_score], pred, inferred


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


def pred_site(df, pred_label, meth_label):
    comb_pred = df.pred_prob.min() + df.pred_prob.max()
    if comb_pred >= 1:
        pred_label.append(1)
    else: 
        pred_label.append(0)
    meth_label.append(df.methyl_label.unique()[0])

    return pred_label, meth_label


def pred_site_deepmod(df, pred_label, meth_label, threshold=0.3):
    inferred = df['inferred_label'].values
    if np.sum(inferred) / len(inferred) >= threshold:
        pred_label.append(1)
    else: 
        pred_label.append(0)
    meth_label.append(df.methyl_label.unique()[0])

    return pred_label, meth_label


def get_accuracy_pos(meth_label, pred_label):
    pos = np.argwhere(np.asarray(meth_label) == 1)
    neg = np.argwhere(np.asarray(meth_label) == 0)

    pred_pos = np.asarray(pred_label)[pos]
    pred_neg = np.asarray(pred_label)[neg]

    accuracy = (sum(pred_pos) + len(pred_neg) - sum(pred_neg)) / \
        (len(pred_pos) + len(pred_neg)) 
    return accuracy[0]


def do_per_position_analysis(df, output):
    df['id'] = df['chrom'] + '_' + df['pos'].astype(str)
    meth_label = []; pred_label = []; cov = []; new_df = pd.DataFrame()
    pred_label_cov = []
    for i, j in df.groupby('id'):
        if len(j.methyl_label.unique()) > 1:
            for k, l in j.groupby('methyl_label'):
                if len(l) > 0:
                    pred_label, meth_label = pred_site(l, pred_label, meth_label)
                    cov.append(len(l))
        else:
            if len(j) > 0:
                pred_label, meth_label = pred_site(j, pred_label, meth_label)
                cov.append(len(j))
            
    precision, recall, f_score, _ = precision_recall_fscore_support(
        meth_label, pred_label, average='binary'
    )

    pl.accuracy_cov(pred_label, meth_label, cov, output)
    # TODO generalize for test with no label 
    # TODO improve calling of a methylation
    # TODO Add to the joint analysis 
    # TODO delete all unnecessary functions
    accuracy = get_accuracy_pos(meth_label, pred_label)
    ut.save_output(
        [accuracy, precision, recall, f_score], output, 'position_accuracy.txt'
    )


def preprocess_error(data, bases):
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    embedding_size = 5
    embedded_bases = tf.one_hot(bases, embedding_size)
    
    size_feat = int(data.shape[1] / 5)

    return tf.concat([embedded_bases, tf.reshape(data, [-1, 5, size_feat])], axis=2)


def call_mods(model, test_file, model_err, model_seq, one_hot_embedding, 
    kmer_sequence, output, figures=False):

    if model == 'seq':

        if test_file.rsplit('.')[-1] == 'tsv':
            test = pd.read_csv(test_file, sep='\t', nrows=1100000,
                names=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 
                'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_lens', 
                'cent_signals', 'methyl_label']) 
            pr.preprocess_sequence(test, os.path.dirname(test_file), 'test')
            test_file = os.path.join(os.path.dirname(test_file), 'test_seq.h5')

        data_seq, labels = ut.get_data_sequence(
            test_file, kmer_sequence, one_hot_embedding
        )

        acc, pred, inferred = acc_test_single(data_seq, labels, model_seq)
        ut.save_probs(pred, labels, output)
        try:
            test['pred_prob'] = pred; test['inferred_label'] = inferred
            pl.plot_distributions(test, output)
            do_per_position_analysis(test, output)
        except: 
            print('No position analysis performed. Only per-read accuracy run')

    elif model == 'err':

        data_err, labels, bases = ut.load_error_data(test_file)
        data_err = preprocess_error(data_err, bases)
        acc, pred, inferred = acc_test_single(data_err, labels, model_err)
        ut.save_probs(pred, labels, output)

    elif model == 'joint':

        data_seq, labels_seq = ut.get_data_sequence(
            test_file, kmer_sequence, one_hot_embedding
        )
        data_err, labels_err = ut.load_error_data(test_file)
 
        acc, probs = acc_test_joint(data_seq, labels_seq, model_seq, data_err, 
            labels_err, model_err)
        
        labels = labels_seq
        ut.save_probs(probs, labels, output)
        
    ut.save_output(acc, output, 'accuracy_measurements.txt')
    
    
    if figures:
        out_fig = os.path.join(output, 'ROC_curve.png')
        pl.plot_ROC(labels, probs, out_fig)

