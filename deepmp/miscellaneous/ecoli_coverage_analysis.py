#!/usr/bin/env python3 

import os
import click
import numpy as np
import pandas as pd
from scipy.special import gamma
from sklearn.metrics import precision_recall_fscore_support

epsilon = 0.05
gamma_val = 0.8
beta_a = 1
beta_b = 22
beta_c = 14.5
# beta_a = 1
# beta_b = 6.5
# beta_c = 10.43

names_deepsignal = ['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 't/c', 'pred_unmod', 'pred_prob', 'inferred_label', 'kmer']

def beta_fct(a, b):
        return gamma(a) * gamma(b) / gamma(a + b)


def likelihood_nomod_beta(obs_reads):
    return np.prod(obs_reads ** (beta_a - 1) * (1 - obs_reads) ** (beta_b - 1) \
        * (1 - epsilon) / beta_fct(beta_a, beta_b) \
        + obs_reads ** (beta_c - 1) * (1 - obs_reads) ** (beta_a - 1) \
        * epsilon / beta_fct(beta_c, beta_a))


def likelihood_mod_beta(obs_reads):
    return np.prod(obs_reads ** (beta_a - 1) * (1 - obs_reads) ** (beta_b - 1) \
        * gamma_val / beta_fct(beta_a, beta_b) \
        + obs_reads ** (beta_c - 1) * (1 - obs_reads) ** (beta_a - 1) \
        * (1 - gamma_val) / beta_fct(beta_c, beta_a))


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


def do_beta_analysis(df, pred_beta):
    pred_beta = beta_stats(df['pred_prob'].values, pred_beta)

    return pred_beta


def pred_site_all(df, pred_005, pred_01, pred_02, pred_03, pred_04, pred_05, label):
        
    for i in [(pred_005, 0.05), (pred_01, 0.1), (pred_02, 0.2), (pred_03, 0.3), (pred_04, 0.4), (pred_05, 0.5)]:
        inferred = df['inferred_label'].values
        if np.sum(inferred) / len(inferred) >= i[1]:
            i[0].append(1)
        else:
            i[0].append(0)

    if len(df.methyl_label.unique()) == 2:
        meth_label.append(1)
    else:
        meth_label.append(df.methyl_label.unique()[0])
    
    return pred_005, pred_01, pred_02, pred_03, pred_04, pred_05, label


#We could sample the coverages more than once. 
def do_cov_analysis(df, cov_1x, label_1x, cov_2x, label_2x, cov_5x, label_5x, cov_10x, label_10x,
    pred_005_1x, pred_01_1x, pred_02_1x, pred_03_1x, pred_04_1x, pred_05_1x,
    pred_005_2x, pred_01_2x, pred_02_2x, pred_03_2x, pred_04_2x, pred_05_2x,
    pred_005_5x, pred_01_5x, pred_02_5x, pred_03_5x, pred_04_5x, pred_05_5x,
    pred_005_10x, pred_01_10x, pred_02_10x, pred_03_10x, pred_04_10x, pred_05_10x):
    df_shape = df.shape[0]

    if len(df.methyl_label.unique()) == 2:
        overall_label = 1
    else:
        overall_label = df.methyl_label.unique()[0]

    x1 = df.sample(n=1)
    cov_1x = do_beta_analysis(x1, cov_1x)
    pred_005_1x, pred_01_1x, pred_02_1x, pred_03_1x, pred_04_1x, pred_05_1x = \
        pred_site_all(x1, pred_005_1x, pred_01_1x, pred_02_1x, 
            pred_03_1x, pred_04_1x, pred_05_1x)
    label_1x.append(label) 

    if df_shape >= 2:
        x2 = df.sample(n=2)
        cov_2x = do_beta_analysis(x2, cov_2x)
        pred_005_2x, pred_01_2x, pred_02_2x, pred_03_2x, pred_04_2x, pred_05_2x = \
            pred_site_all(x2, pred_005_2x, pred_01_2x, pred_02_2x, 
                pred_03_2x, pred_04_2x, pred_05_2x)
        label_2x.append(label) 
    
    if df_shape >= 5: 
        x5 = df.sample(n=5)
        cov_5x = do_beta_analysis(x5, cov_5x)
        pred_005_5x, pred_01_5x, pred_02_5x, pred_03_5x, pred_04_5x, pred_05_5x = \
            pred_site_all(x5, pred_005_5x, pred_01_5x, pred_02_5x, 
                pred_03_5x, pred_04_5x, pred_05_5x)
        label_5x.append(label) 

    if df_shape >= 60:
        x10 = df.sample(n=60)
        cov_10x = do_beta_analysis(x10, cov_10x)
        pred_005_10x, pred_01_10x, pred_02_10x, pred_03_10x, pred_04_10x, pred_05_10x = \
            pred_site_all(x10, pred_005_10x, pred_01_10x, pred_02_10x, 
                pred_03_10x, pred_04_10x, pred_05_10x)
        label_10x.append(label)
    
    # import pdb; pdb.set_trace()
    return cov_1x, label_1x, cov_2x, label_2x, cov_5x, label_5x, cov_10x, label_10x, \
        pred_005_1x, pred_01_1x, pred_02_1x, pred_03_1x, pred_04_1x, pred_05_1x, \
        pred_005_2x, pred_01_2x, pred_02_2x, pred_03_2x, pred_04_2x, pred_05_2x, \
        pred_005_5x, pred_01_5x, pred_02_5x, pred_03_5x, pred_04_5x, pred_05_5x, \
        pred_005_10x, pred_01_10x, pred_02_10x, pred_03_10x, pred_04_10x, pred_05_10x


def get_ids(path):

    df = pd.read_csv(path, sep='\t')
    df['id_per_read'] = df['chrom'] + '_' + df['pos'].astype(str) + '_' + df['readname']

    return df[['id_per_read', 'methyl_label']]

# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='Separate mixid and single CpG human')
@click.option(
    '-tdm', '--test_deepmp', default='',
    help='deepmp test to separate'
)
@click.option(
    '-tdm', '--test_deepmod', default='',
    help='deepmod test to separate'
)
@click.option(
    '-tds', '--test_deepsignal', default='',
    help='deepsignal test.tsv to separate'
)
@click.option(
    '-o', '--output', default='', 
    help='Output file extension'
)
def main(test_deepmp, test_deepsignal, test_deepmod, output):

    ids = get_ids(test_deepmp)

    if test_deepmp:
        deepmp = pd.read_csv(test_deepmp, sep='\t')
    
    if test_deepsignal:
        test = pd.read_csv(test_deepsignal, sep='\t', header=None, names=names_deepsignal).drop_duplicates()
        test['id_per_read'] = test['chrom'] + '_' + test['pos'].astype(str) + '_' + test['readname'] 
        test['id'] = test['chrom'] + '_' + test['pos'].astype(str)
        deepsignal = pd.merge(test, ids, on='id_per_read', how='inner')

    if test_deepmod:
        deepmod = pd.read_csv(test_deepmod, sep='\t')
        deepmod['inferred_label'] = deepmod['mod_pred']
        deepmod['pred_prob'] = deepmod['mod_pred']

    
    import pdb;pdb.set_trace()
    cov_1x, cov_2x, cov_5x, cov_10x = [], [], [], []
    label_1x, label_2x, label_5x, label_10x = [], [], [], []
    pred_005_1x, pred_01_1x, pred_02_1x, pred_03_1x, pred_04_1x, pred_05_1x = [], [], [], [], [], []
    pred_005_2x, pred_01_2x, pred_02_2x, pred_03_2x, pred_04_2x, pred_05_2x = [], [], [], [], [], []
    pred_005_5x, pred_01_5x, pred_02_5x, pred_03_5x, pred_04_5x, pred_05_5x = [], [], [], [], [], []
    pred_005_10x, pred_01_10x, pred_02_10x, pred_03_10x, pred_04_10x, pred_05_10x = [], [], [], [], [], []

    counter = 0
    for i, j in deepmp.groupby('id'):
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

        if counter % 10000 == 0:
            print(precision_recall_fscore_support(label_1x, cov_1x,  average='binary'))
            print(precision_recall_fscore_support(label_2x, cov_2x, average='binary'))
            print(precision_recall_fscore_support(label_5x, cov_5x, average='binary'))
            print(precision_recall_fscore_support(label_10x, cov_10x,  average='binary'))
            print(precision_recall_fscore_support(label_1x, pred_01_1x,  average='binary'))
            print(precision_recall_fscore_support(label_2x, pred_01_2x,  average='binary'))
            print(precision_recall_fscore_support(label_5x, pred_01_5x,  average='binary'))
            print(precision_recall_fscore_support(label_10x, pred_01_10x,  average='binary'))
            # print(precision_recall_fscore_support(label_10x, pred_01_10x,  average='binary'))
            # print(round(1 - np.argwhere(np.asarray(label_1x) != np.asarray(cov_1x)).shape[0] / len(np.asarray(label_1x)), 5))
            # print(round(1 - np.argwhere(np.asarray(label_2x) != np.asarray(cov_2x)).shape[0] / len(np.asarray(label_2x)), 5))
            # print(round(1 - np.argwhere(np.asarray(label_5x) != np.asarray(cov_5x)).shape[0] / len(np.asarray(label_5x)), 5))
            # print(round(1 - np.argwhere(np.asarray(label_10x) != np.asarray(cov_10x)).shape[0] / len(np.asarray(label_10x)), 5))
            # print(round(1 - np.argwhere(np.asarray(label_1x) != np.asarray(pred_01_1x)).shape[0] / len(np.asarray(label_1x)), 5))
            # print(round(1 - np.argwhere(np.asarray(label_1x) != np.asarray(pred_01_2x)).shape[0] / len(np.asarray(label_1x)), 5))
            # print(round(1 - np.argwhere(np.asarray(label_1x) != np.asarray(pred_01_5x)).shape[0] / len(np.asarray(label_1x)), 5))
            # print(round(1 - np.argwhere(np.asarray(label_1x) != np.asarray(pred_01_10x)).shape[0] / len(np.asarray(label_1x)), 5))
            # print(round(1 - np.argwhere(np.asarray(label_1x) != np.asarray(pred_01_1x)).shape[0] / len(np.asarray(label_1x)), 5))
            print()
    

if __name__ == "__main__":
    main()