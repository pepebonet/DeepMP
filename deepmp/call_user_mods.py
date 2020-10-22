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


def test_single_read(data, model_file, score_av='binary'):
    model = load_model(model_file)
    
    pred =  model.predict(data).flatten()
    inferred = np.zeros(len(pred), dtype=int)
    inferred[np.argwhere(pred >= 0.5)] = 1

    return pred, inferred


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


def do_per_position_analysis(df, pred_vec, inferred_vec, output, pred_type):
    df['id'] = df['chrom'] + '_' + df['pos'].astype(str)
    df['pred_prob'] = pred_vec
    df['inferred_label'] = inferred_vec
    meth_label = []; pred_label = []; cov = []; new_df = pd.DataFrame()
    pred_label_cov = []

    import pdb;pdb.set_trace()
    for i, j in df.groupby('id'):
        pred_label, meth_label = pred_site(j, pred_label, meth_label, pred_type)
        cov.append(len(j))
        import pdb;pdb.set_trace()


def build_test_df(data):
    df = pd.DataFrame()
    df['chrom'] = data[0].astype(str)
    df['pos'] = data[2]
    df['strand'] = data[3].astype(str)
    df['pos_in_strand'] = data[4]
    df['readname'] = data[1].astype(str)
    
    return df


#TODO remove labels
def call_mods_user(model_type, test_file, trained_model, kmer, output,
                    err_features = False, pos_based = False ,
                    pred_type = 'min_max', figures=False):

    ## process text file input
    if test_file.rsplit('.')[-1] == 'tsv':
        print("processing tsv file, this might take a while...")
        test = pd.read_csv(test_file, sep='\t', names=pr.names_all)
        ut.preprocess_combined(test, os.path.dirname(test_file), '', 'test_all')
        test_file = os.path.join(os.path.dirname(test_file), 'test_all.h5')

    ## read-based calling
    if model_type == 'seq':
        data_seq, labels = ut.get_data_sequence(test_file, kmer, err_features)
        pred, inferred = test_single_read(data_seq, trained_model)

    elif model_type == 'err':
        data_err, labels = ut.get_data_errors(test_file, kmer)
        pred, inferred = test_single_read(data_err, trained_model)

    elif model_type == 'joint':
        data_seq, data_err, labels, data_id = ut.get_data_jm(test_file, kmer, get_id=True)
        import pdb;pdb.set_trace()
        pred, inferred = test_single_read([data_seq, data_err], trained_model)

    else:
        print("unrecognized model type")
        return None

    ut.save_probs_user(pred, inferred, output)

    ## position-based calling
    # TODO store position info in test file
    if pos_based:
        if 'data_id' in locals():
            test = build_test_df(data_id)
            #TODO DELETE
            test['methyl_label'] = labels

        do_per_position_analysis(test, pred, inferred, output, pred_type)