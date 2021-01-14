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
read0_pos1 = 0.80
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
def beta_stats(obs_reads, pred_beta, prob_beta_mod, prob_beta_unmod):
    prob_pos_0 = likelihood_nomod_beta(obs_reads)
    prob_pos_1 = likelihood_mod_beta(obs_reads)
    
    prob_beta_mod.append(prob_pos_1 / (prob_pos_0 + prob_pos_1))
    prob_beta_unmod.append(prob_pos_0 / (prob_pos_0 + prob_pos_1))

    if prob_beta_mod[-1] >= prob_beta_unmod[-1]:
        pred_beta.append(1)
    else:
        pred_beta.append(0)

    return pred_beta, prob_beta_mod, prob_beta_unmod


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
    test = pd.read_csv(test_file, sep='\t')

    meth_label = []
    pred_beta = []
    prob_beta_mod = []
    prob_beta_unmod = []

    for i, j in test.groupby('id'):
        pred_beta, prob_beta_mod, prob_beta_unmod = beta_stats(
            j['pred_prob'].values, pred_beta, prob_beta_mod, prob_beta_unmod
        )

        if len(j.methyl_label.unique()) == 2:
            meth_label.append(1)
        else:
            meth_label.append(j.methyl_label.unique()[0])

    
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()
