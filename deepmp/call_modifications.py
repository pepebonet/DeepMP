#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support
from deepmp.model import *
import deepmp.utils as ut
# import deepmp.plots as pl #this script needs to debug
import deepmp.preprocess as pr


def acc_test_single(data, labels, model, score_av='binary'):
    #model = load_model(model_file)
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

# TODO remove predifined threshold
def pred_site(df, pred_label, meth_label,
                pred_type, threshold=0.3):

    ## min+max prediction
    if pred_type == 'min_max':
        comb_pred = df.pred_prob.min() + df.pred_prob.max()
        if comb_pred >= 1:
            pred_label.append(1)
        else:
            pred_label.append(0)
        meth_label.append(df.methyl_label.unique()[0])

    ## threshold prediction
    elif pred_type =='threshold':
        inferred = df['inferred_label'].values
        if np.sum(inferred) / len(inferred) >= threshold:
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


def do_per_position_analysis(df, pred_vec ,inferred_vec ,output, pred_type):
    df['id'] = df['chrom'] + '_' + df['pos'].astype(str)
    df['pred_prob'] = pred_vec
    df['inferred_label'] = inferred_vec
    meth_label = []; pred_label = []; cov = []; new_df = pd.DataFrame()
    pred_label_cov = []
    for i, j in df.groupby('id'):
        if len(j.methyl_label.unique()) > 1:
            for k, l in j.groupby('methyl_label'):
                if len(l) > 0:
                    pred_label, meth_label = pred_site(l, pred_label, meth_label, pred_type)
                    cov.append(len(l))
        else:
            if len(j) > 0:
                pred_label, meth_label = pred_site(j, pred_label, meth_label, pred_type)
                cov.append(len(j))

    precision, recall, f_score, _ = precision_recall_fscore_support(
        meth_label, pred_label, average='binary'
    )

    # pl.accuracy_cov(pred_label, meth_label, cov, output)
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


def call_mods(model_type, test_file, trained_model, kmer, output,
                    err_features = False, pos_based = False ,
                    pred_type = 'min_max', figures=False):

    ## process text file input
    if test_file.rsplit('.')[-1] == 'tsv':
        print("processing tsv file, this might take a while...")
        test = pd.read_csv(test_file, sep='\t', names=pr.names_all)
        ut.preprocess_combined(test, os.path.dirname(test_file), 'all', 'test')
        test_file = os.path.join(os.path.dirname(test_file), 'test_all.h5')

    ## read-based calling
    if model_type == 'seq':

        data_seq, labels = ut.get_data_sequence(test_file, kmer, err_features)
        try:
            model = load_model(trained_model)
        except:
            model = SequenceCNN('conv', 6, 256, 4)
            input_shape = (None, kmer, 9)
            model.compile(loss='binary_crossentropy',
                                  optimizer=tf.keras.optimizers.Adam(),
                                  metrics=['accuracy'])
            model.build(input_shape)
            model.load_weights(trained_model)
        acc, pred, inferred = acc_test_single(data_seq, labels, model)

    elif model_type == 'err':

        data_err, labels = ut.get_data_errors(test_file, kmer)
        try:
            model = load_model(trained_model)
        except:
            model = BCErrorCNN(3, 3, 128, 3)
            model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
            input_shape = (None, kmer, 9)
            model.build(input_shape)
            model.load_weights(trained_model)
        acc, pred, inferred = acc_test_single(data_err, labels, model)

    elif model_type == 'joint':
        data_seq, data_err, labels = ut.get_data_jm(test_file, kmer)
        try:
            model = load_model(trained_model)
        except:
            model = JointNN()
            model.compile(loss='binary_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(learning_rate=0.00125),
                           metrics=['accuracy'])
            input_shape = ([(None, kmer, 9), (None, kmer, 9)])
            model.build(input_shape)
            model.load_weights(trained_model)
        acc, pred, inferred = acc_test_single([data_seq, data_err], labels, model)

    else:
        print("unrecognized model type")
        return None

    ut.save_probs(pred, labels, output)
    ut.save_output(acc, output, 'accuracy_measurements.txt')

    ## position-based calling
    # TODO store position info in test file
    if pos_based:
        #pl.plot_distributions(test, output)
        do_per_position_analysis(test, pred, inferred, output, pred_type)

    #if figures:
    #    out_fig = os.path.join(output, 'ROC_curve.png')
    #    pl.plot_ROC(labels, probs, out_fig)

    return None
