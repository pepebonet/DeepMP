#!/usr/bin/env python3 

import os
import click
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import gamma
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import precision_recall_fscore_support

epsilon = 0.05
gamma_val = 0.8
beta_a = 1
beta_b = 22
beta_c = 14.5
# beta_a = 1
# beta_b = 6.5
# beta_c = 10.43

palette_dict = { 
        'DeepMP':'#08519c',
        'DeepSignal':'#f03b20', 
        'DeepMod':'#238443'
}

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


def do_beta_analysis(df, pred_beta, label):
    pred_beta = beta_stats(df['pred_prob'].values, pred_beta)

    if len(df.methyl_label.unique()) == 2:
        label.append(1)
    else:
        label.append(df.methyl_label.unique()[0])

    return pred_beta, label


def get_sample(df, n, percentage):
    meth = df[df['methyl_label'] == 1]
    unmeth = df[df['methyl_label'] == 0]
    try:
        meth_sample = meth.sample(int(n * (percentage / 100)))
        unmeth_sample = unmeth.sample(int(n * (1 - percentage / 100)))
    except: 
        try:
            meth_sample = meth.sample(int(n * (percentage / 100)), replace=True)
            unmeth_sample = unmeth.sample(int(n * (1 - percentage / 100)), replace=True)
        except:
            import pdb;pdb.set_trace()

    return pd.concat([meth_sample, unmeth_sample])


#We could sample the coverages more than once. 
def do_cov_analysis(df, cov_1x, label_1x, cov_2x, label_2x, cov_5x, label_5x, 
    cov_10x, label_10x, cov_20x, label_20x, cov_30x, label_30x):
    df_shape = df.shape[0]

    if len(df['methyl_label'].unique()) != 1:
        x1 = df.sample(n=1)
        cov_1x, label_1x = do_beta_analysis(x1, cov_1x, label_1x) 

        if df_shape >= 2:
            x2 = df.sample(n=2)
            cov_2x, label_2x = do_beta_analysis(x2, cov_2x, label_2x)
        
        if df_shape >= 5: 
            # x5 = get_sample(df, 5, 20)
            x5 = df.sample(n=5)
            cov_5x, label_5x = do_beta_analysis(x5, cov_5x, label_5x)

        if df_shape >= 10:
            x10 = get_sample(df, 10, 20) 
            cov_10x, label_10x = do_beta_analysis(x10, cov_10x, label_10x)

        if df_shape >= 20:
            x20 = get_sample(df, 20, 20)
            cov_20x, label_20x = do_beta_analysis(x20, cov_20x, label_20x)

        if df_shape >= 30:
            x30 = get_sample(df, 30, 20)
            cov_30x, label_30x = do_beta_analysis(x30, cov_30x, label_30x)
    
    return cov_1x, label_1x, cov_2x, label_2x, cov_5x, label_5x, \
        cov_10x, label_10x, cov_20x, label_20x, cov_30x, label_30x


def get_ids(path):

    df = pd.read_csv(path, sep='\t')
    df['id_per_read'] = df['chrom'] + '_' + df['pos'].astype(str) + '_' + df['readname']

    return df[['id_per_read', 'methyl_label']]


def extract_preds(df):
    cov_1x, cov_2x, cov_5x, cov_10x, cov_20x, cov_30x = [], [], [], [], [], []
    label_1x, label_2x, label_5x, label_10x, label_20x, label_30x = [], [], [], [], [], []

    counter = 0
    for i, j in df.groupby('id'):
        cov_1x, label_1x, cov_2x, label_2x, cov_5x, label_5x, cov_10x, label_10x, \
            cov_20x, label_20x, cov_30x, label_30x  = \
            do_cov_analysis(
                j, cov_1x, label_1x, cov_2x, label_2x, cov_5x, label_5x, 
                cov_10x, label_10x, cov_20x, label_20x, cov_30x, label_30x
        )

    prec_1x, rec_1x, f_sco_1x, _ = precision_recall_fscore_support(label_1x, cov_1x,  average='binary')
    prec_2x, rec_2x, f_sco_2x, _ = precision_recall_fscore_support(label_2x, cov_2x, average='binary')
    prec_5x, rec_5x, f_sco_5x, _ = precision_recall_fscore_support(label_5x, cov_5x, average='binary')
    prec_10x, rec_10x, f_sco_10x, _ = precision_recall_fscore_support(label_10x, cov_10x,  average='binary')
    prec_20x, rec_20x, f_sco_20x, _ = precision_recall_fscore_support(label_20x, cov_20x,  average='binary')
    prec_30x, rec_30x, f_sco_30x, _ = precision_recall_fscore_support(label_30x, cov_30x,  average='binary')
    
    return (cov_1x, label_1x, cov_2x, label_2x, cov_5x, label_5x, cov_10x, label_10x , cov_20x, label_20x, cov_30x, label_30x), \
        [f_sco_1x, f_sco_2x, f_sco_5x, f_sco_10x, f_sco_20x, f_sco_30x]


def plot_f_scores(deepmp, deepsignal, deepmod, output):

    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    positions = np.array([1, 2, 3, 4, 5, 6])

    custom_lines = []
    for pred in [(deepmp, 'DeepMP'), (deepsignal, 'DeepSignal'), (deepmod, 'DeepMod')]:
        custom_lines.append(
            plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
            mew=1, color=palette_dict[pred[1]], label=pred[1])[0] 
        )

        if pred[1] == 'DeepMP':
            pos = -0.2
        elif pred[1] == 'DeepMod':
            pos = 0.2
        else:
            pos = 0

        plt.plot(positions + pos, pred[0], marker="o", ms=7, ls="", mec='black', 
            mew=1, color=palette_dict[pred[1]], label=pred[1])[0] 


    ax.set_xlabel("Coverage", fontsize=12)
    ax.set_ylabel("F-score", fontsize=12)
    import pdb;pdb.set_trace()
    # ax.set_xticklabels([1, 2, 5, 10, 20, 30])
    ax.set_xticklabels(['0', '1', '2', '5', '10', '20', '30'])
    plt.xticks(rotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    plt.tight_layout()
    out_dir = os.path.join(output, 'coverage_analysis_20.pdf')
    plt.savefig(out_dir)
    plt.close()


def plot_f_scores_barplot(deepmp, deepsignal, deepmod, output):

    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 5), facecolor='white', gridspec_kw={'height_ratios':[7,1]})

    ax.set_ylim(.76, 1.) 
    ax2.set_ylim(0, .12)

    cov = ['1x', '2x', '5x', '10x', '20x', '30x']
    deepMP = pd.DataFrame([deepmp, ['DeepMP']*6, cov]).T
    deepSignal = pd.DataFrame([deepsignal, ['DeepSignal']*6, cov]).T
    deepMod = pd.DataFrame([deepmod, ['DeepMod']*6, cov]).T
    df = pd.concat([deepMP, deepSignal, deepMod])

    sns.barplot(x=2, y=0, hue=1, data=df, ax=ax, palette=['#08519c', '#f03b20', '#238443'], \
        hue_order=['DeepMP', 'DeepSignal', 'DeepMod'])
    sns.barplot(x=2, y=0, hue=1, data=df, ax=ax2, palette=['#08519c', '#f03b20', '#238443'], \
        hue_order=['DeepMP', 'DeepSignal', 'DeepMod'])


    custom_lines = []
    for pred in [(deepmp, 'DeepMP'), (deepsignal, 'DeepSignal'), (deepmod, 'DeepMod')]:
        custom_lines.append(
            plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
            mew=1, color=palette_dict[pred[1]], label=pred[1])[0] 
        )

    ax.set_ylabel("F-score", fontsize=12)
    ax.set_xlabel("", fontsize=12)
    ax2.set_ylabel("", fontsize=12)
    ax2.set_xlabel("Coverage", fontsize=12)

    plt.xticks(rotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax.tick_params(bottom = False)
    ax.tick_params(labelbottom = False)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    d = .01  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    ax2.get_legend().remove()

    plt.tight_layout()
    out_dir = os.path.join(output, 'coverage_analysis_20_barplot.pdf')
    plt.savefig(out_dir)
    plt.close()



# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='Separate mixid and single CpG human')
@click.option(
    '-tdm', '--test_deepmp', default='',
    help='deepmp test to separate'
)
@click.option(
    '-tdmo', '--test_deepmod', default='',
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

    preds_deepmp, f_score_deepmp = extract_preds(deepmp)
    import pdb;pdb.set_trace()
    preds_deepsignal, f_score_deepsignal = extract_preds(deepsignal)
    import pdb;pdb.set_trace()
    preds_deepmod, f_score_deepmod = extract_preds(deepmod)
    import pdb;pdb.set_trace()

    # plot_f_scores(f_score_deepmp, f_score_deepsignal, f_score_deepmod, output)
    plot_f_scores_barplot(f_score_deepmp, f_score_deepsignal, f_score_deepmod, output)
    
    

if __name__ == "__main__":
    main()
