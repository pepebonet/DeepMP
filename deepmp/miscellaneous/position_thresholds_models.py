#!/usr/bin/env python3
import os
import click
import seaborn
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import gamma
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import precision_recall_fscore_support

read1_pos0 = 0.005  
read0_pos1 = 0.9 
fp = 0.001  
fn = 0.001  
beta_a = 1
beta_b = 5

palette_dict = { 
        'DeepMP':'#08519c',
        'DeepSignal':'#f03b20', 
        'DeepMod':'#238443'
}


names_deepsignal = ['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 't/c', 'pred_unmod', 'pred_prob', 'inferred_label', 'kmer']

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


def do_per_position_analysis(df):

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


def get_ids(paths):
    ids = []
    for path in paths: 
        df = pd.read_csv(path, sep='\t')
        df['id_per_read'] = df['chrom'] + '_' + df['pos'].astype(str) + '_' + df['readname']
        ids.append((df[['id_per_read', 'methyl_label']], path.rsplit('/', 1)[1].split('_')[0]))

    return ids


def extract_preds_deepmp(predictions_file, output):
    q1, median, q3 = [], [], []
    q1_fp, median_fp, q3_fp = [], [], []
    q1_fn, median_fn, q3_fn = [], [], []
    label = []
    for file in tqdm(predictions_file):
        test = pd.read_csv(file, sep='\t').drop_duplicates()
        label.append(file.rsplit('/', 1)[1].split('_')[0])
        # import pdb;pdb.set_trace()
        all_preds = do_per_position_analysis(test)
        Q1, med, Q3 = np.percentile(all_preds['meth_freq_diff'].values, [25, 50, 75])
        q1.append(Q1); median.append(med); q3.append(Q3)
        Q1_fp, med_fp, Q3_fp = np.percentile(all_preds['fp_freq'].values, [25, 50, 75])
        q1_fp.append(Q1_fp); median_fp.append(med_fp); q3_fp.append(Q3_fp)
        Q1_fn, med_fn, Q3_fn = np.percentile(all_preds['fn_freq'].values, [25, 50, 75])
        q1_fn.append(Q1_fn); median_fn.append(med_fn); q3_fn.append(Q3_fn)
    
    return (q1, median, q3, label, 'DeepMP')


def extract_preds_deepsignal(predictions_file, ids):
    q1, median, q3 = [], [], []
    q1_fp, median_fp, q3_fp = [], [], []
    q1_fn, median_fn, q3_fn = [], [], []
    label = []
    for file in tqdm(predictions_file):
        mod_perc = file.rsplit('/', 1)[1].split('_')[2]
        for el in ids: 
            if el[1] == mod_perc:
                id_labels = el[0]
                break

        test = pd.read_csv(file, sep='\t', header=None, names=names_deepsignal).drop_duplicates()
        test['id_per_read'] = test['chrom'] + '_' + test['pos'].astype(str) + '_' + test['readname'] 
        test['id'] = test['chrom'] + '_' + test['pos'].astype(str)
        test = pd.merge(test, id_labels, on='id_per_read', how='inner')
        # import pdb;pdb.set_trace()
        label.append(file.rsplit('/', 1)[1].split('_')[2])

        all_preds = do_per_position_analysis(test)
        Q1, med, Q3 = np.percentile(all_preds['meth_freq_diff'].values, [25, 50, 75])
        q1.append(Q1); median.append(med); q3.append(Q3)
        Q1_fp, med_fp, Q3_fp = np.percentile(all_preds['fp_freq'].values, [25, 50, 75])
        q1_fp.append(Q1_fp); median_fp.append(med_fp); q3_fp.append(Q3_fp)
        Q1_fn, med_fn, Q3_fn = np.percentile(all_preds['fn_freq'].values, [25, 50, 75])
        q1_fn.append(Q1_fn); median_fn.append(med_fn); q3_fn.append(Q3_fn)
    
    return (q1, median, q3, label, 'DeepSignal')


def extract_preds_deepmod(predictions_file, output):
    q1, median, q3 = [], [], []
    q1_fp, median_fp, q3_fp = [], [], []
    q1_fn, median_fn, q3_fn = [], [], []
    label = []
    for file in tqdm(predictions_file):
        test = pd.read_csv(file, sep='\t')
        test['inferred_label'] = test['mod_pred']
        test['pred_prob'] = test['mod_pred']
        # import pdb;pdb.set_trace()
        label.append(file.rsplit('/', 2)[1].split('_')[0])

        all_preds = do_per_position_analysis(test)
        Q1, med, Q3 = np.percentile(all_preds['meth_freq_diff'].values, [25, 50, 75])
        q1.append(Q1); median.append(med); q3.append(Q3)
        Q1_fp, med_fp, Q3_fp = np.percentile(all_preds['fp_freq'].values, [25, 50, 75])
        q1_fp.append(Q1_fp); median_fp.append(med_fp); q3_fp.append(Q3_fp)
        Q1_fn, med_fn, Q3_fn = np.percentile(all_preds['fn_freq'].values, [25, 50, 75])
        q1_fn.append(Q1_fn); median_fn.append(med_fn); q3_fn.append(Q3_fn)
    
    return (q1, median, q3, label, 'DeepMod')


def plot_please(Q1, median, Q3, label, output):

    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')

    yerr = np.array([[np.asarray(median) - np.asarray(Q1)], [np.asarray(Q3) - np.asarray(median)]])
    yerr = yerr.reshape(2, yerr.shape[2])

    plt.errorbar(
        np.asarray(label).astype(np.int), np.asarray(median), yerr=yerr, 
        marker='o', mfc='#08519c', mec='black', ms=10, mew=1, 
        ecolor='black', capsize=2.5, elinewidth=1, capthick=1
    )

    ax.set_xlabel("Methylation percentage", fontsize=12)
    ax.set_ylabel("Inferred Frequency - True Frequency", fontsize=12)
    plt.xticks(rotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.ylim(top=0.1, bottom=-0.01)
    plt.xlim(left=-1, right=102)

    plt.tight_layout()
    out_dir = os.path.join(output, 'methylation_frequency.pdf')
    plt.savefig(out_dir)
    plt.close()


def plot_comparison(deepmp, deepsignal, deepmod, output):

    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')

    custom_lines = []
    for pred in [deepmp, deepsignal, deepmod]:
        custom_lines.append(
            plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
            mew=1, color=palette_dict[pred[4]], label=pred[4])[0] 
        )

        if pred[4] == 'DeepMP':
            pos = -2.5
        elif pred[4] == 'DeepMod':
            pos = 2.5
        else:
            pos = 0

        yerr = np.array([[np.asarray(pred[1]) - np.asarray(pred[0])], [np.asarray(pred[2]) - np.asarray(pred[1])]])
        yerr = yerr.reshape(2, yerr.shape[2])

        plt.errorbar(
            np.asarray(pred[3]).astype(np.int) + pos, np.asarray(pred[1]), yerr=yerr, 
            marker='o', mfc=palette_dict[pred[4]], mec='black', ms=10, mew=1, 
            ecolor='black', capsize=2.5, elinewidth=1, capthick=1
        )

    ax.set_xlabel("Methylation percentage", fontsize=12)
    ax.set_ylabel("Inferred Frequency - True Frequency", fontsize=12)
    plt.xticks(rotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plt.ylim(top=0.1, bottom=-0.01)
    plt.xlim(left=-1, right=102)

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    plt.tight_layout()
    out_dir = os.path.join(output, 'methylation_frequency.pdf')
    plt.savefig(out_dir)
    plt.close()


@click.command(short_help='Convert DeepMod output into accuracy scores.')
@click.option(
    '-pd', '--predictions_deepmp', multiple=True,
    help='Folder containing the predictions on test from deepmp'
)
@click.option(
    '-pds', '--predictions_deepsignal', multiple=True,
    help='Folder containing the predictions on test from deepsignal'
)
@click.option(
    '-pdm', '--predictions_deepmod', multiple=True,
    help='Folder containing the predictions on test from deepmod'
)
@click.option(
    '-o', '--output', default='', help='output folder'
)
def main(predictions_deepmp, predictions_deepsignal, predictions_deepmod, output):
    ids = get_ids(predictions_deepmp)
    preds_deepmp = extract_preds_deepmp(predictions_deepmp, output)
    preds_deepsignal = extract_preds_deepsignal(predictions_deepsignal, ids)
    preds_deepmod = extract_preds_deepmod(predictions_deepmod, output)
    
    # plot_please(q1, median, q3, label, output)
    plot_comparison(preds_deepmp, preds_deepsignal, preds_deepmod, output)
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()