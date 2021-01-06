#!/usr/bin/env python3
import os
import click
import seaborn
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy.special import gamma
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import precision_recall_fscore_support

read1_pos0 = 0.02  
read0_pos1 = 0.80
fp = 0.02  
fn = 0.03  
beta_a = 1
beta_b = 8

palette_dict = { 
        'DeepMP':'#08519c',
        'DeepSignal':'#f03b20', 
        'DeepMod':'#238443'
}
palette_dict_2 = {
        'DeepMP_1':'#08519c', 'DeepMP_2':'#6baed6', 
        'DeepMP_3':'#a6bddb', 'DeepMP_4':'#f1eef6'
}


names_deepsignal = ['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 't/c', 'pred_unmod', 'pred_prob', 'inferred_label', 'kmer']

def pred_site_all(df, pred_min_max, pred_001, pred_002, pred_004, pred_005, pred_006, 
    pred_008, pred_009, pred_01, pred_02, pred_03, pred_04,  meth_label):

    ## min+max prediction
    comb_pred = df.pred_prob.min() + df.pred_prob.max()
    if comb_pred >= 1:
        pred_min_max.append(1)
    else:
        pred_min_max.append(0)
        
    ## threshold prediction
    for i in [(pred_001, 0.01), (pred_002, 0.02), (pred_004, 0.04), (pred_005, 0.05), (pred_006, 0.06), \
        (pred_008, 0.08), (pred_009, 0.09), (pred_01, 0.1), (pred_02, 0.2), (pred_03, 0.3), (pred_04, 0.4)]:
        inferred = df['inferred_label'].values
        if np.sum(inferred) / len(inferred) >= i[1]:
            i[0].append(1)
        else:
            i[0].append(0)
    
    if len(df.methyl_label.unique()) == 2:
        meth_label.append(1)
    else:
        meth_label.append(df.methyl_label.unique()[0])
    
    return pred_min_max, pred_001, pred_002, pred_004, pred_005, pred_006, pred_008, \
        pred_009, pred_01, pred_02, pred_03, pred_04, meth_label


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


def pred_deepsignal_pos(df, pred_deepsignal):
    if df['pred_prob'].sum() / len(df) >= 0.5: 
        pred_deepsignal.append(1)
    else: 
        pred_deepsignal.append(0)
    
    return pred_deepsignal


def do_per_position_analysis(df):

    cov = []; pred_min_max = []; pred_005 = []; pred_009 = []; pred_01 = []; pred_02 = []
    pred_03 = []; pred_04 = []; meth_label = []; ids = []; pred_posterior = []
    prob_mod = []; prob_unmod = []; pred_beta = []; prob_beta_mod = []
    prob_beta_unmod = []; meth_freq_diff = []; fp_freq = []; fn_freq = []
    true_meth_freq = []; pred_deepsignal = []; 
    pred_001, pred_002, pred_004, pred_006, pred_008 = [], [], [], [], []

    for i, j in df.groupby('id'):
        meth_freq_diff.append(1 - (np.absolute(j.inferred_label.values - j.methyl_label).sum() / len(j)))
        true_meth_freq.append(j.methyl_label.sum() / len(j))
        fp_freq.append(j.inferred_label.values[np.argwhere(j.methyl_label.values == 0)].sum() \
            / len(np.argwhere(j.methyl_label.values == 0)))
        fn_freq.append((len(np.argwhere(j.methyl_label.values == 1)) - \
            j.inferred_label.values[np.argwhere(j.methyl_label.values == 1)].sum()) \
            / len(np.argwhere(j.methyl_label.values == 1)))

        pred_deepsignal = pred_deepsignal_pos(j, pred_deepsignal)

        pred_min_max, pred_001, pred_002, pred_004, pred_005, pred_006, pred_008,\
            pred_009, pred_01, pred_02, pred_03, pred_04, meth_label = pred_site_all(
            j, pred_min_max,  pred_001, pred_002, pred_004, pred_005, 
            pred_006, pred_008, pred_009, pred_01, pred_02, pred_03, 
            pred_04, meth_label
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
    preds['pred_001'] = pred_001
    preds['pred_002'] = pred_002
    preds['pred_004'] = pred_004
    preds['pred_005'] = pred_005
    preds['pred_006'] = pred_006
    preds['pred_008'] = pred_008
    preds['pred_009'] = pred_009
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
    preds['pred_deepsignal'] = pred_deepsignal
    preds['meth_label'] = meth_label 
    preds['meth_freq_diff'] = meth_freq_diff
    preds['true_meth_freq'] = true_meth_freq
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
    label = []; accuracy_beta, accuracy_01, accuracy_005, accuracy_001 = [], [], [], []
    for file in tqdm(predictions_file):
        test = pd.read_csv(file, sep='\t').drop_duplicates()
        label.append(file.rsplit('/', 1)[1].split('_')[0])
        # import pdb;pdb.set_trace()
        all_preds = do_per_position_analysis(test)
        
        labels = all_preds['meth_label'].values
        inf_beta = all_preds['pred_beta'].values
        inf_01 = all_preds['pred_01'].values
        inf_005 = all_preds['pred_005'].values
        inf_001 = all_preds['pred_001'].values

        acc_beta = round(1 - np.argwhere(labels != inf_beta).shape[0] / len(labels), 5)
        acc_01 = round(1 - np.argwhere(labels != inf_01).shape[0] / len(labels), 5)
        acc_005 = round(1 - np.argwhere(labels != inf_005).shape[0] / len(labels), 5)
        acc_001 = round(1 - np.argwhere(labels != inf_001).shape[0] / len(labels), 5)

        accuracy_beta.append(acc_beta); accuracy_01.append(acc_01)
        accuracy_005.append(acc_005); accuracy_001.append(acc_001)
        
        Q1, med, Q3 = np.percentile(all_preds['meth_freq_diff'].values, [25, 50, 75])
        q1.append(Q1); median.append(med); q3.append(Q3)
        Q1_fp, med_fp, Q3_fp = np.percentile(all_preds['fp_freq'].values, [25, 50, 75])
        q1_fp.append(Q1_fp); median_fp.append(med_fp); q3_fp.append(Q3_fp)
        Q1_fn, med_fn, Q3_fn = np.percentile(all_preds['fn_freq'].values, [25, 50, 75])
        q1_fn.append(Q1_fn); median_fn.append(med_fn); q3_fn.append(Q3_fn)
    
    return (q1, median, q3, label, 'DeepMP', accuracy_beta, accuracy_beta, \
        accuracy_01, accuracy_005, accuracy_001)


def extract_preds_deepsignal(predictions_file, ids):
    q1, median, q3 = [], [], []
    q1_fp, median_fp, q3_fp = [], [], []
    q1_fn, median_fn, q3_fn = [], [], []
    label = []; accuracy_dps = []; accuracy_beta = []
    accuracy_01, accuracy_005, accuracy_001 = [], [], []

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

        labels = all_preds['meth_label'].values
        inf_dps = all_preds['pred_deepsignal'].values
        inf_beta = all_preds['pred_beta'].values
        inf_01 = all_preds['pred_01'].values
        inf_005 = all_preds['pred_005'].values
        inf_001 = all_preds['pred_001'].values
        acc_dps = round(1 - np.argwhere(labels != inf_dps).shape[0] / len(labels), 5)
        acc_beta = round(1 - np.argwhere(labels != inf_beta).shape[0] / len(labels), 5)
        acc_01 = round(1 - np.argwhere(labels != inf_01).shape[0] / len(labels), 5)
        acc_005 = round(1 - np.argwhere(labels != inf_005).shape[0] / len(labels), 5)
        acc_001 = round(1 - np.argwhere(labels != inf_001).shape[0] / len(labels), 5)
        accuracy_dps.append(acc_dps); accuracy_beta.append(acc_beta)
        accuracy_01.append(acc_01)
        accuracy_005.append(acc_005); accuracy_001.append(acc_001)

        Q1, med, Q3 = np.percentile(all_preds['meth_freq_diff'].values, [25, 50, 75])
        q1.append(Q1); median.append(med); q3.append(Q3)
        Q1_fp, med_fp, Q3_fp = np.percentile(all_preds['fp_freq'].values, [25, 50, 75])
        q1_fp.append(Q1_fp); median_fp.append(med_fp); q3_fp.append(Q3_fp)
        Q1_fn, med_fn, Q3_fn = np.percentile(all_preds['fn_freq'].values, [25, 50, 75])
        q1_fn.append(Q1_fn); median_fn.append(med_fn); q3_fn.append(Q3_fn)
    
    return (q1, median, q3, label, 'DeepSignal', accuracy_beta, \
        accuracy_dps, accuracy_01, accuracy_005, accuracy_001)


def extract_preds_deepmod(predictions_file, output):
    q1, median, q3 = [], [], []
    q1_fp, median_fp, q3_fp = [], [], []
    q1_fn, median_fn, q3_fn = [], [], []
    label = []; accuracy_beta = []; accuracy_02 = []
    accuracy_01, accuracy_005, accuracy_001 = [], [], []

    for file in tqdm(predictions_file):
        test = pd.read_csv(file, sep='\t')
        test['inferred_label'] = test['mod_pred']
        test['pred_prob'] = test['mod_pred']
        # import pdb;pdb.set_trace()
        label.append(file.rsplit('/', 2)[1].split('_')[0])

        all_preds = do_per_position_analysis(test)

        labels = all_preds['meth_label'].values
        inf_02 = all_preds['pred_02'].values
        inf_beta = all_preds['pred_beta'].values
        inf_01 = all_preds['pred_01'].values
        inf_005 = all_preds['pred_005'].values
        inf_001 = all_preds['pred_001'].values

        acc_02 = round(1 - np.argwhere(labels != inf_02).shape[0] / len(labels), 5)
        acc_beta = round(1 - np.argwhere(labels != inf_beta).shape[0] / len(labels), 5)
        acc_01 = round(1 - np.argwhere(labels != inf_01).shape[0] / len(labels), 5)
        acc_005 = round(1 - np.argwhere(labels != inf_005).shape[0] / len(labels), 5)
        acc_001 = round(1 - np.argwhere(labels != inf_001).shape[0] / len(labels), 5)

        accuracy_02.append(acc_02); accuracy_beta.append(acc_beta)
        accuracy_01.append(acc_01)
        accuracy_005.append(acc_005); accuracy_001.append(acc_001)

        Q1, med, Q3 = np.percentile(all_preds['meth_freq_diff'].values, [25, 50, 75])
        q1.append(Q1); median.append(med); q3.append(Q3)
        Q1_fp, med_fp, Q3_fp = np.percentile(all_preds['fp_freq'].values, [25, 50, 75])
        q1_fp.append(Q1_fp); median_fp.append(med_fp); q3_fp.append(Q3_fp)
        Q1_fn, med_fn, Q3_fn = np.percentile(all_preds['fn_freq'].values, [25, 50, 75])
        q1_fn.append(Q1_fn); median_fn.append(med_fn); q3_fn.append(Q3_fn)
    
    return (q1, median, q3, label, 'DeepMod', accuracy_beta, \
        accuracy_02, accuracy_01, accuracy_005, accuracy_001)


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
    sns.set_theme(style="darkgrid", palette = "Set2")
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')

    custom_lines = []
    for pred in [deepmp, deepsignal, deepmod]:
        custom_lines.append(
            plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
            mew=1, color=palette_dict[pred[4]], label=pred[4])[0] 
        )

        if pred[4] == 'DeepMP':
            pos = -3
        elif pred[4] == 'DeepMod':
            pos = 3
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
    ax.set_ylabel("1 - (Inferred Frequency - True Frequency)", fontsize=12)
    plt.xticks(rotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plt.ylim(bottom=0.5)
    # plt.xlim(left=-1, right=102)

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    plt.tight_layout()
    out_dir = os.path.join(output, 'methylation_frequency.pdf')
    plt.savefig(out_dir)
    plt.close()


def plot_pos_accuracy(deepmp, deepsignal, deepmod, output):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')

    custom_lines = []
    for pred in [deepmp, deepsignal, deepmod]:
        custom_lines.append(
            plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
            mew=1, color=palette_dict[pred[4]], label=pred[4])[0] 
        )

        if pred[4] == 'DeepMP':
            pos = -3
            plt.plot(
                np.asarray(pred[3]).astype(np.int)[1:] + pos, pred[5][1:], 
                marker='o', mfc=palette_dict[pred[4]], mec='black', ms=10, mew=1, 
                c=palette_dict[pred[4]]
            )
        elif pred[4] == 'DeepMod':
            pos = 3
            plt.plot(
                np.asarray(pred[3]).astype(np.int)[1:] + pos, pred[6][1:], 
                marker='o', mfc=palette_dict[pred[4]], mec='black', ms=10, mew=1, 
                c=palette_dict[pred[4]]
            )
        else:
            pos = 0
            plt.plot(
                np.asarray(pred[3]).astype(np.int)[1:] + pos, pred[6][1:], 
                marker='o', mfc=palette_dict[pred[4]], mec='black', ms=10, mew=1, 
                c=palette_dict[pred[4]]
            )
    import pdb;pdb.set_trace()

    ax.set_xlabel("Methylation percentage", fontsize=12)
    ax.set_ylabel("Position Accuracy", fontsize=12)
    plt.xticks(rotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    plt.tight_layout()
    out_dir = os.path.join(output, 'position_accuracy.pdf')
    plt.savefig(out_dir)
    plt.close()


def plot_pos_accuracy_beta(deepmp, deepsignal, deepmod, output):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')

    custom_lines = []
    for pred in [deepmp, deepsignal, deepmod]:
        custom_lines.append(
            plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
            mew=1, color=palette_dict[pred[4]], label=pred[4])[0] 
        )

        if pred[4] == 'DeepMP':
            pos = -3
            plt.plot(
                np.asarray(pred[3]).astype(np.int)[1:] + pos, pred[5][1:], 
                marker='o', mfc=palette_dict[pred[4]], mec='black', ms=10, mew=1, 
                c=palette_dict[pred[4]]
            )
        elif pred[4] == 'DeepMod':
            pos = 3
            plt.plot(
                np.asarray(pred[3]).astype(np.int)[1:] + pos, pred[5][1:], 
                marker='o', mfc=palette_dict[pred[4]], mec='black', ms=10, mew=1, 
                c=palette_dict[pred[4]]
            )
        else:
            pos = 0
            plt.plot(
                np.asarray(pred[3]).astype(np.int)[1:] + pos, pred[5][1:], 
                marker='o', mfc=palette_dict[pred[4]], mec='black', ms=10, mew=1, 
                c=palette_dict[pred[4]]
            )
    import pdb;pdb.set_trace()

    ax.set_xlabel("Methylation percentage", fontsize=12)
    ax.set_ylabel("Position Accuracy", fontsize=12)
    plt.xticks(rotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.ylim(bottom=0, top=1.05)

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    plt.tight_layout()
    out_dir = os.path.join(output, 'position_accuracy_beta.pdf')
    plt.savefig(out_dir)
    plt.close()


def plot_pos_accuracy_around_0(deepmp, deepsignal, deepmod, output):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    import pdb;pdb.set_trace()
    labels = ['Beta Model', '10{} Filter'.format('%'), '5{} Filter'.format('%'), '1{} Filter'.format('%')]
    positions = [0.2, -0.2, -0.05, 0.05]
    custom_lines = []
    for pred in [deepmp]:
        

        if pred[4] == 'DeepMP':
            
            counter = 1
            pos = -0.1
            for el in pred[5:9]:
                import pdb;pdb.set_trace()
                custom_lines.append(
                    plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
                    mew=1, color=palette_dict_2[pred[4] + '_' + str(counter)], label=labels[counter - 1])[0] 
                )
                plt.plot(
                    np.asarray(pred[3]).astype(np.int) + positions[counter - 1], el, 
                    marker='o', mfc=palette_dict_2[pred[4] + '_' + str(counter)], mec='black', 
                    ms=8, mew=1, c=palette_dict_2[pred[4] + '_' + str(counter)]
                )

                counter += 1 
        elif pred[4] == 'DeepMod':

            pos = 0.1
            for el in pred[7:10]:
                plt.plot(
                    np.asarray(pred[3]).astype(np.int) + pos, el, 
                    marker='o', mfc=palette_dict[pred[4]], mec='black', ms=10, mew=1, 
                    c=palette_dict[pred[4]]
                )
        else:
            pos = 0
            for el in pred[7:10]:
                plt.plot(
                    np.asarray(pred[3]).astype(np.int) + pos, el, 
                    marker='o', mfc=palette_dict[pred[4]], mec='black', ms=10, mew=1, 
                    c=palette_dict[pred[4]]
                )
    import pdb;pdb.set_trace()

    ax.set_xlabel("Methylation percentage", fontsize=12)
    ax.set_ylabel("Position Accuracy", fontsize=12)
    plt.xticks(rotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.ylim(bottom=0, top=1.05)

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    plt.tight_layout()
    out_dir = os.path.join(output, 'position_accuracy_around0.pdf')
    plt.savefig(out_dir)
    plt.close()


def plot_pos_barplot_around_0(deepmp, deepsignal, deepmod, output):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')

    df = pd.DataFrame()
    custom_lines = []

    for pred in [deepmp, deepsignal, deepmod]:
        custom_lines.append(
            plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
            mew=1, color=palette_dict[pred[4]], label=pred[4])[0] 
        )

        alg_df = pd.DataFrame(
            [[pred[4], pred[5][0], 'Beta Model'],[pred[4], pred[7][0], '10% Filter'], \
            [pred[4], pred[8][0], '5% Filter'], [pred[4], pred[9][0], '1% Filter']], \
            columns=['Model', 'Accuracy', 'Filter']
        )
        df = pd.concat([df, alg_df])


    sns.barplot(x="Filter", y="Accuracy", hue="Model", data=df, 
        palette=['#08519c', '#f03b20','#238443'])

    ax.set_xlabel("Filters", fontsize=12)
    ax.set_ylabel("Position Accuracy", fontsize=12)
    plt.xticks(rotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.ylim(bottom=0, top=1.05)

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    plt.tight_layout()
    out_dir = os.path.join(output, 'position_accuracy_around0_barplot.pdf')
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
    '-az', '--around_zero', is_flag=True, default=False, 
    help='whether the inputs are around 0 percentages'
)
@click.option(
    '-o', '--output', default='', help='output folder'
)
def main(predictions_deepmp, predictions_deepsignal, predictions_deepmod, around_zero, output):
    ids = get_ids(predictions_deepmp)

    preds_deepmp = extract_preds_deepmp(predictions_deepmp, output)
    preds_deepsignal = extract_preds_deepsignal(predictions_deepsignal, ids)
    preds_deepmod = extract_preds_deepmod(predictions_deepmod, output)
    # import pdb;pdb.set_trace()
    # plot_please(q1, median, q3, label, output)
    # plot_comparison(preds_deepmp, preds_deepsignal, preds_deepmod, output)

    if around_zero:
        plot_pos_barplot_around_0(preds_deepmp, preds_deepsignal, preds_deepmod, output)
        # plot_pos_accuracy_around_0(preds_deepmp, preds_deepsignal, preds_deepmod, output)
        
    
    else:
        plot_pos_accuracy(preds_deepmp, preds_deepsignal, preds_deepmod, output)
        plot_pos_accuracy_beta(preds_deepmp, preds_deepsignal, preds_deepmod, output)
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()