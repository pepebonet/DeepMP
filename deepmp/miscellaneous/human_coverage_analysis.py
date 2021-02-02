#!/usr/bin/env python3

import os 
import sys
import click
import itertools
import numpy as np
import pandas as pd
from itertools import islice
from collections import Counter
from scipy.special import gamma
from sklearn.metrics import precision_recall_fscore_support

sys.path.append('../')
import deepmp.utils as ut

read1_pos0 = 0.02
read0_pos1 = 0.8
beta_a = 1
beta_b = 8

names_all=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname',
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_skew', 'signal_kurt', 'signal_diff', 'signal_lens',
            'cent_signals', 'qual', 'mis', 'ins', 'del', 'methyl_label']


def beta_fct(a, b):
        return gamma(a) * gamma(b) / gamma(a + b)


def likelihood_nomod_beta(obs_reads):
    return np.prod(obs_reads ** (beta_a - 1) * (1 - obs_reads) ** (beta_b - 1) \
        * (1 - read1_pos0) / beta_fct(beta_a, beta_b) \
        + obs_reads ** (beta_b - 1) * (1 - obs_reads) ** (beta_a - 1) \
        * read1_pos0 / beta_fct(beta_b, beta_a))


def likelihood_mod_beta(obs_reads):
    return np.prod(obs_reads ** (beta_a - 1) * (1 - obs_reads) ** (beta_b - 1) \
        * read0_pos1 / beta_fct(beta_a, beta_b) \
        + obs_reads ** (beta_b - 1) * (1 - obs_reads) ** (beta_a - 1) \
        * (1 - read0_pos1) / beta_fct(beta_b, beta_a))


#Assuming prior to be 0.5
def beta_stats(obs_reads, pred_beta):
    prob_pos_0 = likelihood_nomod_beta(obs_reads)
    prob_pos_1 = likelihood_mod_beta(obs_reads)

    prob_beta_mod = prob_pos_1 / (prob_pos_0 + prob_pos_1)
    prob_beta_unmod = prob_pos_0 / (prob_pos_0 + prob_pos_1)

    if prob_beta_mod >= prob_beta_unmod:
        pred_beta.append(1)
    else:
        pred_beta.append(0)

    return pred_beta


def do_beta_analysis(df, pred_beta, meth_label):
    pred_beta = beta_stats(df['pred_prob'].values, pred_beta)

    if len(df.methyl_label.unique()) == 2:
        meth_label.append(1)
    else:
        meth_label.append(df.methyl_label.unique()[0])
    if meth_label[-1] != pred_beta[-1]:
        import pdb;pdb.set_trace()
    return pred_beta, meth_label


def pred_site_all(df, pred_005, pred_01, pred_02, pred_03, pred_04, pred_05):
        
    for i in [(pred_005, 0.05), (pred_01, 0.1), (pred_02, 0.2), (pred_03, 0.3), (pred_04, 0.4), (pred_05, 0.5)]:
        inferred = df['inferred_label'].values
        if np.sum(inferred) / len(inferred) >= i[1]:
            i[0].append(1)
        else:
            i[0].append(0)
    
    return pred_005, pred_01, pred_02, pred_03, pred_04, pred_05



def do_cov_analysis(df, cov_1x, label_1x, cov_2x, label_2x, cov_5x, label_5x, cov_10x, label_10x,
    pred_005_1x, pred_01_1x, pred_02_1x, pred_03_1x, pred_04_1x, pred_05_1x,
    pred_005_2x, pred_01_2x, pred_02_2x, pred_03_2x, pred_04_2x, pred_05_2x,
    pred_005_5x, pred_01_5x, pred_02_5x, pred_03_5x, pred_04_5x, pred_05_5x,
    pred_005_10x, pred_01_10x, pred_02_10x, pred_03_10x, pred_04_10x, pred_05_10x):
    df_shape = df.shape[0]
    x1 = df.sample(n=1)
    cov_1x, label_1x = do_beta_analysis(x1, cov_1x, label_1x)
    pred_005_1x, pred_01_1x, pred_02_1x, pred_03_1x, pred_04_1x, pred_05_1x = \
        pred_site_all(x1, pred_005_1x, pred_01_1x, pred_02_1x, 
            pred_03_1x, pred_04_1x, pred_05_1x)

    if df_shape >= 2:
        x2 = df.sample(n=2)
        cov_2x, label_2x = do_beta_analysis(x2, cov_2x, label_2x)
        pred_005_2x, pred_01_2x, pred_02_2x, pred_03_2x, pred_04_2x, pred_05_2x = \
            pred_site_all(x2, pred_005_2x, pred_01_2x, pred_02_2x, 
                pred_03_2x, pred_04_2x, pred_05_2x)
    
    if df_shape >= 5: 
        x5 = df.sample(n=5)
        cov_5x, label_5x = do_beta_analysis(x5, cov_5x, label_5x)
        pred_005_5x, pred_01_5x, pred_02_5x, pred_03_5x, pred_04_5x, pred_05_5x = \
            pred_site_all(x5, pred_005_5x, pred_01_5x, pred_02_5x, 
                pred_03_5x, pred_04_5x, pred_05_5x)

    if df_shape >= 10:
        x10 = df.sample(n=10)
        cov_10x, label_10x = do_beta_analysis(x10, cov_10x, label_10x)
        pred_005_10x, pred_01_10x, pred_02_10x, pred_03_10x, pred_04_10x, pred_05_10x = \
            pred_site_all(x10, pred_005_10x, pred_01_10x, pred_02_10x, 
                pred_03_10x, pred_04_10x, pred_05_10x)
    
    return cov_1x, label_1x, cov_2x, label_2x, cov_5x, label_5x, cov_10x, label_10x, \
        pred_005_1x, pred_01_1x, pred_02_1x, pred_03_1x, pred_04_1x, pred_05_1x, \
        pred_005_2x, pred_01_2x, pred_02_2x, pred_03_2x, pred_04_2x, pred_05_2x, \
        pred_005_5x, pred_01_5x, pred_02_5x, pred_03_5x, pred_04_5x, pred_05_5x, \
        pred_005_10x, pred_01_10x, pred_02_10x, pred_03_10x, pred_04_10x, pred_05_10x


# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='Separate mixid and single CpG human')
@click.option(
    '-tf', '--test_file', required=True,
    help='test.tsv to separate'
)
@click.option(
    '-o', '--output', default='', 
    help='Output file extension'
)
def main(test_file, output):
    test = pd.read_csv(test_file, sep='\t', nrows=5000000)

    cov_1x, cov_2x, cov_5x, cov_10x = [], [], [], []
    label_1x, label_2x, label_5x, label_10x = [], [], [], []
    pred_005_1x, pred_01_1x, pred_02_1x, pred_03_1x, pred_04_1x, pred_05_1x = [], [], [], [], [], []
    pred_005_2x, pred_01_2x, pred_02_2x, pred_03_2x, pred_04_2x, pred_05_2x = [], [], [], [], [], []
    pred_005_5x, pred_01_5x, pred_02_5x, pred_03_5x, pred_04_5x, pred_05_5x = [], [], [], [], [], []
    pred_005_10x, pred_01_10x, pred_02_10x, pred_03_10x, pred_04_10x, pred_05_10x = [], [], [], [], [], []

    counter = 0
    for i, j in test.groupby('id'):
        cov_1x, label_1x, cov_2x, label_2x, cov_5x, label_5x, cov_10x, label_10x, \
        pred_005_1x, pred_01_1x, pred_02_1x, pred_03_1x, pred_04_1x, pred_05_1x, \
        pred_005_2x, pred_01_2x, pred_02_2x, pred_03_2x, pred_04_2x, pred_05_2x, \
        pred_005_5x, pred_01_5x, pred_02_5x, pred_03_5x, pred_04_5x, pred_05_5x, \
        pred_005_10x, pred_01_10x, pred_02_10x, pred_03_10x, pred_04_10x, pred_05_10x = \
            do_cov_analysis(j, cov_1x, label_1x, cov_2x, label_2x, cov_5x, label_5x, cov_10x, label_10x,
            pred_005_1x, pred_01_1x, pred_02_1x, pred_03_1x, pred_04_1x, pred_05_1x,
            pred_005_2x, pred_01_2x, pred_02_2x, pred_03_2x, pred_04_2x, pred_05_2x,
            pred_005_5x, pred_01_5x, pred_02_5x, pred_03_5x, pred_04_5x, pred_05_5x,
            pred_005_10x, pred_01_10x, pred_02_10x, pred_03_10x, pred_04_10x, pred_05_10x
        )

        counter += 1 

        if counter % 1000 == 0:
            # print(precision_recall_fscore_support(cov_1x, label_1x, average='binary'))
            # print(precision_recall_fscore_support(cov_2x, label_2x, average='binary'))
            # print(precision_recall_fscore_support(cov_5x, label_5x, average='binary'))
            print(precision_recall_fscore_support(label_10x, cov_10x,  average='binary'))
            print(precision_recall_fscore_support(label_10x, pred_005_10x,  average='binary'))
            print(precision_recall_fscore_support(label_10x, pred_01_10x,  average='binary'))
            print(precision_recall_fscore_support(label_10x, pred_02_10x,  average='binary'))
            print(precision_recall_fscore_support(label_10x, pred_03_10x,  average='binary'))
            print(precision_recall_fscore_support(label_10x, pred_05_10x,  average='binary'))
            print()
    
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()
