#!/usr/bin/envs python3

import os
import sys
import click
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support

sys.path.append('../')
import deepmp.utils as ut
import deepmp.plots as pl

def get_false_true_calls(df):
    fn = df[(df['labels'] == 1) & (df['inferred'] == 0)]
    fn['calls'] = 'FN'
    fp = df[(df['labels'] == 0) & (df['inferred'] == 1)]
    fp['calls'] = 'FP'
    tn = df[(df['labels'] == 0) & (df['inferred'] == 0)]
    tn['calls'] = 'TN'
    tp = df[(df['labels'] == 1) & (df['inferred'] == 1)]
    tp['calls'] = 'TP'

    return pd.concat([fn, fp, tn, tp])


def get_df_from_np(features, labels, predictions, inferred):
    feat_np = features.numpy()
    feat_pd = pd.DataFrame(feat_np)
    feat_pd['labels'] = labels
    feat_pd['predictions'] = predictions
    feat_pd['inferred'] = inferred

    return get_false_true_calls(feat_pd)


def get_ind_feat_seq(features, labels, predictions, inferred):
    mean = get_df_from_np(features[:, :, 5], labels, predictions, inferred) 
    std = get_df_from_np(features[:, :, 6], labels, predictions, inferred)
    median = get_df_from_np(features[:, :, 7], labels, predictions, inferred)  
    rang = get_df_from_np(features[:, :, 8], labels, predictions, inferred)
    len_sig = get_df_from_np(features[:, :, 9], labels, predictions, inferred)

    return mean, median, std, rang, len_sig



def get_ind_feat_err(features, labels, predictions, inferred):
    quality = get_df_from_np(features[:, :, 5], labels, predictions, inferred)
    mismatch = get_df_from_np(features[:, :, 6], labels, predictions, inferred)
    deletion = get_df_from_np(features[:, :, 7], labels, predictions, inferred)
    insertion = get_df_from_np(features[:, :, 8], labels, predictions, inferred) 

    return quality, mismatch, deletion, insertion



def acc_test_single(data, labels, model_file, score_av='binary'):
    model = load_model(model_file)
    
    pred =  model.predict(data).flatten()
    inferred = np.zeros(len(pred), dtype=int)
    inferred[np.argwhere(pred >= 0.5)] = 1

    precision, recall, f_score, _ = precision_recall_fscore_support(
        labels, inferred, average=score_av
    )

    return [precision, recall, f_score], pred, inferred


@click.command(short_help='Explore how features affect our model')
@click.option(
    '-f', '--features', help='features path', required=True
)
@click.option(
    '-m', '--model_type', required=True,
    type=click.Choice(['seq', 'err', 'joint']),
    help='choose model to test'
)
@click.option(
    '-md', '--model', required=True, help='trained model'
)
@click.option(
    '-k', '--kmer', default=17, help='kmer length'
)
@click.option(
    '-o', '--output', help='output path'
)
def main(features, model_type, model, kmer, output):

    if model_type == 'seq':
        data_seq, labels = ut.get_data_sequence(features, kmer, err_features)
        acc, pred, inferred = acc_test_single(data_seq, labels, model)

    elif model_type == 'err':
        data_err, labels = ut.get_data_errors(features, kmer)
        acc, pred, inferred = acc_test_single(data_err, labels, model)

    elif model_type == 'joint':
        data_seq, data_err, labels = ut.get_data_jm(features, kmer)

        acc, pred, inferred = acc_test_single([data_seq, data_err], labels, model)

        mean, median, std, rang, len_sig = get_ind_feat_seq(
            data_seq, labels, pred, inferred
        )
        quality, mismatch, deletion, insertion = get_ind_feat_err(
            data_err, labels, pred, inferred
        )

        to_plot = [(mean, 'mean.pdf'), (median, 'median.pdf'), \
            (std, 'std.pdf'), (rang, 'rang.pdf'), (len_sig, 'len_sig.pdf'), \
            (quality, 'quality.pdf')]
        
        plot_err = [(mismatch, 'mismatch.pdf'), (deletion, \
                'deletion.pdf'), (insertion, 'insertion.pdf')]

        for el in to_plot:
            pl.feature_exploration_plots(el[0], kmer, output, el[1])



if __name__ == "__main__":
    main()