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


def save_output(acc, output):
    col_names = ['Accuracy', 'Precision', 'Recall', 'F-score']
    df = pd.DataFrame([acc], columns=col_names)
    df.to_csv(os.path.join(output, 'accuracy_measurements.txt'), index=False, sep='\t')


def save_output_positions(acc, output):
    col_names = ['Accuracy', 'Precision', 'Recall', 'F-score']
    df = pd.DataFrame([acc], columns=col_names)
    df.to_csv(os.path.join(output, 'accuracy_measurements_positions.txt'), index=False, sep='\t')


def save_probs(probs, labels, output):
    out_probs = os.path.join(output, 'test_pred_prob.txt')
    probs_to_save = pd.DataFrame(columns=['labels', 'probs'])
    probs_to_save['labels'] = labels
    probs_to_save['probs'] = probs
    probs_to_save.to_csv(out_probs, sep='\t', index=None)


def pred_site(df, pred_label, meth_label):
    comb_pred = df.pred_prob.mean()
    if comb_pred >= 0.5:
        pred_label.append(1)
    else: 
        pred_label.append(0)
    meth_label.append(df.methyl_label.unique()[0])

    return pred_label, meth_label


def pred_site_2(df, pred_label, meth_label):
    comb_pred = df.pred_prob.min() + df.pred_prob.max()
    if comb_pred >= 1:
        pred_label.append(1)
    else: 
        pred_label.append(0)
    meth_label.append(df.methyl_label.unique()[0])
    # import pdb;pdb.set_trace()
    return pred_label, meth_label


def option_1(l):
    cc = pd.concat([l[l['pred_prob'] > 0.9], l[l['pred_prob'] < 0.1]])
    comb_pred = cc.pred_prob.mean()


def do_per_position_analysis(df):
    df['id'] = df['chrom'] + '_' + df['pos'].astype(str)
    meth_label = []; pred_label = []
    for i, j in df.groupby('id'):
        if len(j.methyl_label.unique()) > 1:
            for k, l in j.groupby('methyl_label'):
                if len(l) > 0:
                    pred_label, meth_label = pred_site_2(l, pred_label, meth_label)
        else:
            if len(j) > 0:
                pred_label, meth_label = pred_site_2(j, pred_label, meth_label)
            
    precision, recall, f_score, _ = precision_recall_fscore_support(
        meth_label, pred_label, average='binary'
    )
    import pdb; pdb.set_trace()


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
        test['pred_prob'] = pred; test['inferred_label'] = inferred
        do_per_position_analysis(test)

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

