#!/usr/bin/env python3
import os
import click
import numpy as np
import pandas as pd
from scipy.special import gamma
from sklearn.metrics import precision_recall_fscore_support

read1_pos0 = 0.005  
read0_pos1 = 0.9 
fp = 0.001  
fn = 0.001  
beta_a = 1
beta_b = 5

def pred_site_all(df, pred_min_max, pred_005, pred_01, pred_02, pred_03, pred_04,  meth_label):

    ## min+max prediction
    comb_pred = df.pred_prob.min() + df.pred_prob.max()
    if comb_pred >= 1:
        pred_min_max.append(1)
    else:
        pred_min_max.append(0)
        
    ## threshold prediction
    for i in [(pred_005, 0.05), (pred_01, 0.1), (pred_02, 0.2), (pred_03, 0.3), (pred_04, 0.4)]:
        inferred = df['inferred_label'].values
        if np.sum(inferred) / len(inferred) >= i[1]:
            i[0].append(1)
        else:
            i[0].append(0)
    
    if len(df.methyl_label.unique()) == 2:
        meth_label.append(1)
    else:
        meth_label.append(df.methyl_label.unique()[0])
    
    return pred_min_max, pred_005, pred_01, pred_02, pred_03, pred_04, meth_label


#obs_reads is the vector of inferred labels
#fp and fn are false positives and negatives respectively 
#read1_pos0 is the probability of seeing a modified read if the position is called to be 0
#read0_pos1 is the probability of seeing an unmodified read if the position is called to be 1
def likelihood_nomod_pos(obs_reads):
    return np.prod(obs_reads * ((1 - read1_pos0) * fp + read1_pos0 * (1 - fn)) + \
        (1 - obs_reads) * ((1 - read1_pos0) * (1 - fp) + read1_pos0 * fn))


def likelihood_mod_pos(obs_reads):
    return np.prod(obs_reads * (read0_pos1 * fp + (1 - read0_pos1) * (1 - fn)) + \
        (1 - obs_reads) * (read0_pos1 * (1 - fp) + (1 - read0_pos1) * fn))


def pred_stats(obs_reads, pred_posterior, prob_mod, prob_unmod):
    prob_pos_0 = likelihood_nomod_pos(obs_reads)
    prob_pos_1 = likelihood_mod_pos(obs_reads)

    prob_mod.append(prob_pos_1 / (prob_pos_0 + prob_pos_1))
    prob_unmod.append(prob_pos_0 / (prob_pos_0 + prob_pos_1))

    if prob_mod[-1] >= prob_unmod[-1]:
        pred_posterior.append(1)
    else:
        pred_posterior.append(0)

    return pred_posterior, prob_mod, prob_unmod


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


def do_per_position_analysis(df, output):

    cov = []; pred_min_max = []; pred_005 = []; pred_01 = []; pred_02 = []
    pred_03 = []; pred_04 = []; meth_label = []; ids = []; pred_posterior = []
    prob_mod = []; prob_unmod = []; pred_beta = []; prob_beta_mod = []
    prob_beta_unmod = []; meth_freq_diff = []; fp_freq = []; fn_freq = []

    for i, j in df.groupby('id'):
        meth_freq_diff.append(np.absolute(j.inferred_label.values - j.methyl_label).sum() / len(j))
        fp_freq.append(j.inferred_label.values[np.argwhere(j.methyl_label.values == 0)].sum() \
            / len(np.argwhere(j.methyl_label.values == 0)))
        fn_freq.append((len(np.argwhere(j.methyl_label.values == 1)) - \
            j.inferred_label.values[np.argwhere(j.methyl_label.values == 1)].sum()) \
            / len(np.argwhere(j.methyl_label.values == 1)))

        pred_min_max, pred_005, pred_01, pred_02, pred_03, pred_04, meth_label = pred_site_all(
            j, pred_min_max, pred_005, pred_01, pred_02, pred_03, pred_04, meth_label
        )
        pred_posterior, prob_mod, prob_unmod = pred_stats(
            j['inferred_label'].values, pred_posterior, prob_mod, prob_unmod
        )

        pred_beta, prob_beta_mod, prob_beta_unmod = beta_stats(
            j['pred_prob'].values, pred_beta, prob_beta_mod, prob_beta_unmod
        )

        cov.append(len(j)); ids.append(i)

    preds = pd.DataFrame()
    preds['id'] = ids
    preds['cov'] = cov 
    preds['pred_min_max'] = pred_min_max
    preds['pred_005'] = pred_005
    preds['pred_01'] = pred_01
    preds['pred_02'] = pred_02 
    preds['pred_03'] = pred_03 
    preds['pred_04'] = pred_04 
    preds['pred_posterior'] = pred_posterior
    preds['prob_mod'] = prob_mod
    preds['prob_unmod'] = prob_unmod 
    preds['pred_beta'] = pred_beta
    preds['prob_beta_mod'] = prob_mod
    preds['prob_beta_unmod'] = prob_unmod 
    preds['meth_label'] = meth_label 
    preds['meth_freq_diff'] = meth_freq_diff
    preds['fn_freq'] = fn_freq
    preds['fp_freq'] = fp_freq

    return preds

@click.command(short_help='Convert DeepMod output into accuracy scores.')
@click.option(
    '-pf', '--predictions_file', required=True, 
    help='Folder containing the predictions on test'
)
@click.option(
    '-o', '--output', default='', help='output folder'
)
def main(predictions_file, output):
    test = pd.read_csv(predictions_file, sep='\t')
    all_preds = do_per_position_analysis(test, output)
    print(all_preds['meth_freq_diff'].mean(), all_preds['meth_freq_diff'].std())
    print(all_preds['fp_freq'].mean(), all_preds['fp_freq'].std())
    print(all_preds['fn_freq'].mean(), all_preds['fn_freq'].std())
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()