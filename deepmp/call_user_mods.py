#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support

import deepmp.utils as ut
import deepmp.plots as pl
import deepmp.preprocess as pr


def acc_test_single(data, model_file, score_av='binary'):
    model = load_model(model_file)

    pred =  model.predict(data).flatten()
    inferred = np.zeros(len(pred), dtype=int)
    inferred[np.argwhere(pred >= 0.5)] = 1

    return data, pred, inferred


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


def save_output(acc, output, label):
    col_names = ['Accuracy', 'Precision', 'Recall', 'F-score']
    df = pd.DataFrame([acc], columns=col_names)
    df.to_csv(os.path.join(output, label), index=False, sep='\t')


def save_probs(probs, labels, output):
    out_probs = os.path.join(output, 'test_pred_prob.txt')
    probs_to_save = pd.DataFrame(columns=['labels', 'probs'])
    probs_to_save['labels'] = labels
    probs_to_save['probs'] = probs
    probs_to_save.to_csv(out_probs, sep='\t', index=None)


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

    accuracy_cov(pred_label, meth_label, cov, output)
    # TODO generalize for test with no label 
    # TODO improve calling of a methylation
    # TODO Add to the joint analysis 
    # TODO delete all unnecessary functions
    accuracy = get_accuracy_pos(meth_label, pred_label)
    save_output(
        [accuracy, precision, recall, f_score], output, 'position_accuracy.txt'
    )


#TODO send to plots once done
def plot_distributions(df, output):
    fig, ax = plt.subplots(figsize=(5, 5))

    sns.kdeplot(df['pred_prob'], shade=True)
    fig.tight_layout()
    ax.set_xlim(0,1)
    plt.savefig(os.path.join(output, 'distributions.png'))
    plt.close()


def accuracy_cov(pred, label, cov, output):
    df_dict = {'predictions': pred, 'methyl_label': label, 'Coverage': cov}
    df = pd.DataFrame(df_dict)
    cov = []; acc = []

    for i, j in df.groupby('Coverage'):
        cov.append(i)
        acc.append(get_accuracy_pos(
            j['methyl_label'].tolist(), j['predictions'].tolist())
        )
    
    fig, ax = plt.subplots(figsize=(5, 5))

    sns.barplot(cov, acc)
    ax.set_ylim(0.92,1)
    fig.tight_layout()
    
    plt.savefig(os.path.join(output, 'acc_vs_cov.png'))
    plt.close()


def call_user_mods(model, test_file, model_err, model_seq, one_hot_embedding, 
    kmer_sequence, columns, output, figures=False):

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
        data, pred, inferred = acc_test_single(data_seq, labels, model_seq)
        try:
            test['pred_prob'] = pred; test['inferred_label'] = inferred
            plot_distributions(test, output)
            do_per_position_analysis(test, output)
        except: 
            print('No position analysis performed. Only per-read accuracy run')

    elif model == 'err':
        if test_file.rsplit('.', 1)[-1] == 'tsv':
            data_err = pd.read_csv(test_file, sep='\t')
            data_err = ut.select_columns(data_err, columns)
            data_err = data_err.to_numpy().reshape(data_err.shape[0], data_err.shape[1], 1)
        else:
            #TODO <JB> labels cannot be here. An additional script for that. Here only the call of the modifications
            data_err, _ = ut.load_error_data(test_file)
        acc = acc_test_single(data_err, model_err)

    elif model == 'joint':
        data_seq, labels_seq = ut.get_data_sequence(
            test_file, kmer_sequence, one_hot_embedding
        )
        data_err, labels_err = ut.load_error_data(test_file)
 
        acc, probs = acc_test_joint(data_seq, labels_seq, model_seq, data_err, 
            labels_err, model_err)
        
        labels = labels_seq
        save_probs(probs, labels, output)
        
    save_output(acc, output, 'accuracy_measurements.txt')
    
    
    if figures:
        out_fig = os.path.join(output, 'ROC_curve.png')
        pl.plot_ROC(labels, probs, out_fig)
