#!/usr/bin/env python3
import os
import click
import seaborn
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import bottleneck as bn
from scipy.special import gamma
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import precision_recall_fscore_support

epsilon = 0.05  
gamma_val = 0.80
beta_a = 1
beta_b = 20.86
beta_c = 21.25
fp = 0.02  
fn = 0.03  

EPSILON = np.finfo(np.float64).resolution
log_EPSILON = np.log(EPSILON)

palette_dict = { 
        'DeepMP':'#08519c',
        'DeepSignal':'#f03b20', 
        'DeepMod':'#238443',
        'Nanopolish':'#238443',
        'Guppy': '#fed976',
        'Megalodon': '#e7298a'
}
palette_dict_2 = {
        'DeepMP_1':'#08519c', 'DeepMP_2':'#6baed6', 
        'DeepMP_3':'#a6bddb', 'DeepMP_4':'#f1eef6'
}


names_deepsignal = ['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 't/c', 'pred_unmod', 'pred_prob', 'inferred_label', 'kmer']

def pred_site_all(df, pred_001, pred_005, pred_01, pred_02, meth_label):
        
    ## threshold prediction
    for i in [(pred_001, 0.01), (pred_005, 0.05), (pred_01, 0.1), (pred_02, 0.2)]:
        inferred = df['inferred_label'].values
        if np.sum(inferred) / len(inferred) >= i[1]:
            i[0].append(1)
        else:
            i[0].append(0)
    
    if len(df.methyl_label.unique()) == 2:
        meth_label.append(1)
    else:
        meth_label.append(df.methyl_label.unique()[0])
    
    return pred_001, pred_005, \
        pred_01, pred_02, meth_label


#obs_reads is the vector of inferred labels
#fp and fn are false positives and negatives respectively 
#epsilon is the probability of seeing a modified read if the position is called to be 0
#read0_pos1 is the probability of seeing an unmodified read if the position is called to be 1
def likelihood_nomod_pos(obs_reads):
    return np.prod(obs_reads * ((1 - epsilon) * fp + epsilon * (1 - fn)) + \
        (1 - obs_reads) * ((1 - epsilon) * (1 - fp) + epsilon * fn))


def likelihood_mod_pos(obs_reads):
    return np.prod(obs_reads * (gamma_val * fp + (1 - gamma_val) * (1 - fn)) + \
        (1 - obs_reads) * (gamma_val * (1 - fp) + (1 - gamma_val) * fn))


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
        * (1 - epsilon) / beta_fct(beta_a, beta_b) \
        + obs_reads ** (beta_c - 1) * (1 - obs_reads) ** (beta_a - 1) \
        * epsilon / beta_fct(beta_c, beta_a))


def likelihood_mod_beta(obs_reads):
    return np.prod(obs_reads ** (beta_a - 1) * (1 - obs_reads) ** (beta_b - 1) \
        * gamma_val / beta_fct(beta_a, beta_b) \
        + obs_reads ** (beta_c - 1) * (1 - obs_reads) ** (beta_a - 1) \
        * (1 - gamma_val) / beta_fct(beta_c, beta_a))


def log_likelihood_nomod_beta(obs_reads):
    return np.sum(np.log(obs_reads ** (beta_a - 1) * (1 - obs_reads) ** (beta_b - 1) \
        * (1 - epsilon) / beta_fct(beta_a, beta_b) \
        + obs_reads ** (beta_c - 1) * (1 - obs_reads) ** (beta_a - 1) \
        * epsilon / beta_fct(beta_c, beta_a)))


def log_likelihood_mod_beta(obs_reads):
    return np.sum(np.log(obs_reads ** (beta_a - 1) * (1 - obs_reads) ** (beta_b - 1) \
        * gamma_val / beta_fct(beta_a, beta_b) \
        + obs_reads ** (beta_c - 1) * (1 - obs_reads) ** (beta_a - 1) \
        * (1 - gamma_val) / beta_fct(beta_c, beta_a)))


def _normalize_log_probs(probs):
    max_i = bn.nanargmax(probs)
    try:
        exp_probs = np.exp(probs[np.arange(probs.size) != max_i] \
            - probs[max_i])
    except FloatingPointError:
        exp_probs = np.exp(
            np.clip(probs[np.arange(probs.size) != max_i] - probs[max_i],
                log_EPSILON, 0)
        )
    probs_norm = probs - probs[max_i] - np.log1p(bn.nansum(exp_probs))

    return np.exp(np.clip(probs_norm, log_EPSILON, 0))

#Assuming prior to be 0.5
def beta_stats(obs_reads, pred_beta, prob_beta_mod, prob_beta_unmod):

    log_prob_pos_0 = log_likelihood_nomod_beta(obs_reads)
    log_prob_pos_1 = log_likelihood_mod_beta(obs_reads)

    norm_log_probs = _normalize_log_probs(
        np.array([log_prob_pos_0, log_prob_pos_1])
    )

    prob_beta_mod.append(norm_log_probs[1])
    prob_beta_unmod.append(norm_log_probs[0])
    
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


def do_per_position_analysis(df, label):

    cov = []; pred_005 = []; pred_01 = []; pred_02 = []
    meth_label = []; ids = []; pred_beta = []; prob_beta_mod = []
    prob_beta_unmod = []; meth_freq_diff = []; fp_freq = []; fn_freq = []
    true_meth_freq = []; pred_deepsignal = []; pred_001 = []

    for i, j in df.groupby('id'):
            meth_freq = j['inferred_label'].sum() / j.shape[0] * 100
            if meth_freq < int(label[0]) + 10 and meth_freq > int(label[0]) - 10 and j.shape[0] >= 5:
                # import pdb;pdb.set_trace()
                meth_freq_diff.append(1 - (np.absolute(j.inferred_label.values - j.methyl_label).sum() / len(j)))
                true_meth_freq.append(j.methyl_label.sum() / len(j))
                fp_freq.append(j.inferred_label.values[np.argwhere(j.methyl_label.values == 0)].sum() \
                    / len(np.argwhere(j.methyl_label.values == 0)))
                fn_freq.append((len(np.argwhere(j.methyl_label.values == 1)) - \
                    j.inferred_label.values[np.argwhere(j.methyl_label.values == 1)].sum()) \
                    / len(np.argwhere(j.methyl_label.values == 1)))

                pred_deepsignal = pred_deepsignal_pos(j, pred_deepsignal)

                pred_001, pred_005, pred_01, pred_02, meth_label = pred_site_all(
                    j, pred_001, pred_005, pred_01, pred_02, meth_label
                )

                pred_beta, prob_beta_mod, prob_beta_unmod = beta_stats(
                    j['pred_prob'].values, pred_beta, prob_beta_mod, prob_beta_unmod
                )
                
                cov.append(len(j)); ids.append(i)


    preds = pd.DataFrame()
    preds['id'] = ids
    preds['cov'] = cov
    preds['pred_001'] = pred_001
    preds['pred_005'] = pred_005
    preds['pred_01'] = pred_01
    preds['pred_02'] = pred_02 
    preds['pred_beta'] = pred_beta
    preds['prob_beta_mod'] = prob_beta_mod
    preds['prob_beta_unmod'] = prob_beta_unmod 
    preds['pred_deepsignal'] = pred_deepsignal
    preds['meth_label'] = meth_label 
    preds['meth_freq_diff'] = meth_freq_diff
    preds['true_meth_freq'] = true_meth_freq
    preds['fn_freq'] = fn_freq
    preds['fp_freq'] = fp_freq
    #TODO be careful with this. You might need to end up using bigger partial sets
    return preds[preds['cov'] >= 5]


def get_ids(paths):
    ids = []
    for path in paths: 
        df = pd.read_csv(path, sep='\t')
        df['id_per_read'] = df['chrom'] + '_' + df['pos'].astype(str) + '_' + df['readname']
        ids.append((df[['id_per_read', 'methyl_label']], path.rsplit('/')[-2].split('_')[1]))

    return ids


def extract_preds_deepmp(predictions_file):
    q1, median, q3 = [], [], []
    q1_fp, median_fp, q3_fp = [], [], []
    q1_fn, median_fn, q3_fn = [], [], []
    label = []; accuracy_beta, accuracy_01, accuracy_005, accuracy_001 = [], [], [], []

    for file in tqdm(predictions_file):
        test = pd.read_csv(file, sep='\t').drop_duplicates()

        test_min = test[test['pred_prob'] < 0.2]
        test_max = test[test['pred_prob'] > 0.8]
        test = pd.concat([test_min, test_max])

        label.append(file.rsplit('/')[-2].split('_')[1])
        test['id'] = test['chrom'] + '_' + test['strand'] + '_' + test['pos'].astype(str)

        all_preds = do_per_position_analysis(test, label)

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

        if label[-1] != '100':
            Q1_fp, med_fp, Q3_fp = np.percentile(all_preds['fp_freq'].dropna().values, [25, 50, 75])
            q1_fp.append(Q1_fp); median_fp.append(med_fp); q3_fp.append(Q3_fp)
        
        if label[-1] != '0':
            Q1_fn, med_fn, Q3_fn = np.percentile(all_preds['fn_freq'].dropna().values, [25, 50, 75])
            q1_fn.append(Q1_fn); median_fn.append(med_fn); q3_fn.append(Q3_fn)
    
    return (q1, median, q3, label, 'DeepMP', accuracy_beta, accuracy_beta, \
        accuracy_01, accuracy_005, accuracy_001), (q1_fp, median_fp, q3_fp, label, 'DeepMP'), \
            (q1_fn, median_fn, q3_fn, label, 'DeepMP')


def extract_preds_deepsignal(predictions_file, ids):
    q1, median, q3 = [], [], []
    q1_fp, median_fp, q3_fp = [], [], []
    q1_fn, median_fn, q3_fn = [], [], []
    label = []; accuracy_dps = []; accuracy_beta = []
    accuracy_01, accuracy_005, accuracy_001 = [], [], []

    for file in tqdm(predictions_file):
        mod_perc = file.rsplit('/')[-2].split('_')[1]
        for el in ids: 
            if el[1] == mod_perc:
                id_labels = el[0]
                break

        test = pd.read_csv(file, sep='\t', header=None, names=names_deepsignal).drop_duplicates()
        test['id_per_read'] = test['chrom'] + '_' + test['pos'].astype(str) + '_' + test['readname'] 
        test['id'] = test['chrom'] + '_' + test['pos'].astype(str)
        test = pd.merge(test, id_labels, on='id_per_read', how='inner')

        label.append(file.rsplit('/')[-2].split('_')[1])

        all_preds = do_per_position_analysis(test, label)

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

        if label[-1] != '100':
            Q1_fp, med_fp, Q3_fp = np.percentile(all_preds['fp_freq'].dropna().values, [25, 50, 75])
            q1_fp.append(Q1_fp); median_fp.append(med_fp); q3_fp.append(Q3_fp)
        
        if label[-1] != '0':
            Q1_fn, med_fn, Q3_fn = np.percentile(all_preds['fn_freq'].dropna().values, [25, 50, 75])
            q1_fn.append(Q1_fn); median_fn.append(med_fn); q3_fn.append(Q3_fn)
    
    return (q1, median, q3, label, 'DeepSignal', accuracy_beta, \
        accuracy_dps, accuracy_01, accuracy_005, accuracy_001), (q1_fp, median_fp, q3_fp, label, 'DeepSignal'), \
            (q1_fn, median_fn, q3_fn, label, 'DeepSignal')


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
        
        label.append(file.rsplit('/', 2)[1].split('_')[0])

        all_preds = do_per_position_analysis(test, label)

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

        if label[-1] != '100':
            Q1_fp, med_fp, Q3_fp = np.percentile(all_preds['fp_freq'].dropna().values, [25, 50, 75])
            q1_fp.append(Q1_fp); median_fp.append(med_fp); q3_fp.append(Q3_fp)

        if label[-1] != '0':
            Q1_fn, med_fn, Q3_fn = np.percentile(all_preds['fn_freq'].dropna().values, [25, 50, 75])
            q1_fn.append(Q1_fn); median_fn.append(med_fn); q3_fn.append(Q3_fn)
    
    return (q1, median, q3, label, 'DeepMod', accuracy_beta, \
        accuracy_02, accuracy_01, accuracy_005, accuracy_001), (q1_fp, median_fp, q3_fp, label, 'DeepMod'), \
            (q1_fn, median_fn, q3_fn, label, 'DeepMod')


def extract_preds_nanopolish(nanopolish_path, ids):
    q1, median, q3 = [], [], []
    q1_fp, median_fp, q3_fp = [], [], []
    q1_fn, median_fn, q3_fn = [], [], []
    label = []; accuracy_beta = []; accuracy_02 = []
    accuracy_01, accuracy_005, accuracy_001 = [], [], []

    nanopolish = pd.read_csv(nanopolish_path, sep='\t')
    
    nanopolish['inferred_label'] = nanopolish['Prediction']
    nanopolish['pred_prob'] = nanopolish['prob_meth']
    nanopolish['id_per_read'] = nanopolish['chromosome'] + '_' + \
        nanopolish['start'].astype(str)  + '_' + nanopolish['readnames']
    nanopolish['id'] = nanopolish['chromosome'] + '_' + nanopolish['start'].astype(str)

    for file in tqdm(ids):
        label.append(file[1])

        test = pd.merge(file[0], nanopolish, on='id_per_read', how='inner')

        all_preds = do_per_position_analysis(test, label)

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

        if label[-1] != '100':
            Q1_fp, med_fp, Q3_fp = np.percentile(all_preds['fp_freq'].dropna().values, [25, 50, 75])
            q1_fp.append(Q1_fp); median_fp.append(med_fp); q3_fp.append(Q3_fp)

        if label[-1] != '0':
            Q1_fn, med_fn, Q3_fn = np.percentile(all_preds['fn_freq'].dropna().values, [25, 50, 75])
            q1_fn.append(Q1_fn); median_fn.append(med_fn); q3_fn.append(Q3_fn)
    
    return (q1, median, q3, label, 'Nanopolish', accuracy_beta, \
        accuracy_02, accuracy_01, accuracy_005, accuracy_001), (q1_fp, median_fp, q3_fp, label, 'Nanopolish'), \
            (q1_fn, median_fn, q3_fn, label, 'Nanopolish')


def extract_preds_guppy(guppy_path, ids):
    q1, median, q3 = [], [], []
    q1_fp, median_fp, q3_fp = [], [], []
    q1_fn, median_fn, q3_fn = [], [], []
    label = []; accuracy_beta = []; accuracy_02 = []
    accuracy_01, accuracy_005, accuracy_001 = [], [], []

    guppy = pd.read_csv(guppy_path, sep='\t')

    guppy['inferred_label'] = guppy['Prediction']
    guppy['pred_prob'] = guppy['prob_meth']
    guppy['id_per_read'] = guppy['#chromosome'] + '_' + \
        guppy['start'].astype(str)  + '_' + guppy['readnames']
    guppy['id'] = guppy['#chromosome'] + '_' + guppy['start'].astype(str)

    for file in tqdm(ids):
        label.append(file[1])

        test = pd.merge(file[0], guppy, on='id_per_read', how='inner')

        all_preds = do_per_position_analysis(test, label)

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

        if label[-1] != '100':
            Q1_fp, med_fp, Q3_fp = np.percentile(all_preds['fp_freq'].dropna().values, [25, 50, 75])
            q1_fp.append(Q1_fp); median_fp.append(med_fp); q3_fp.append(Q3_fp)

        if label[-1] != '0':
            Q1_fn, med_fn, Q3_fn = np.percentile(all_preds['fn_freq'].dropna().values, [25, 50, 75])
            q1_fn.append(Q1_fn); median_fn.append(med_fn); q3_fn.append(Q3_fn)
    
    return (q1, median, q3, label, 'Guppy', accuracy_beta, \
        accuracy_02, accuracy_01, accuracy_005, accuracy_001), (q1_fp, median_fp, q3_fp, label, 'Guppy'), \
            (q1_fn, median_fn, q3_fn, label, 'Guppy')


def extract_preds_megalodon(megalodon_path, ids):
    q1, median, q3 = [], [], []
    q1_fp, median_fp, q3_fp = [], [], []
    q1_fn, median_fn, q3_fn = [], [], []
    label = []; accuracy_beta = []; accuracy_02 = []
    accuracy_01, accuracy_005, accuracy_001 = [], [], []

    megalodon = pd.read_csv(megalodon_path, sep='\t', header=None)
    meg_pos = megalodon[megalodon[7] > 0.8]
    meg_neg = megalodon[megalodon[7] < 0.2]
    megalodon = pd.concat([meg_pos, meg_neg])

    megalodon['inferred_label']  = megalodon[7].apply(lambda x: 1 if x > 0.5 else 0)
    megalodon['pred_prob'] = megalodon[7]
    megalodon['id_per_read'] = megalodon[1] + '_' + \
        megalodon[3].astype(str)  + '_' + megalodon[9]
    megalodon['id'] = megalodon[1] + '_' + megalodon[3].astype(str)

    for file in tqdm(ids):
        label.append(file[1])

        test = pd.merge(file[0], megalodon, on='id_per_read', how='inner')

        all_preds = do_per_position_analysis(test, label)

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

        if label[-1] != '100':
            Q1_fp, med_fp, Q3_fp = np.percentile(all_preds['fp_freq'].dropna().values, [25, 50, 75])
            q1_fp.append(Q1_fp); median_fp.append(med_fp); q3_fp.append(Q3_fp)

        if label[-1] != '0':
            Q1_fn, med_fn, Q3_fn = np.percentile(all_preds['fn_freq'].dropna().values, [25, 50, 75])
            q1_fn.append(Q1_fn); median_fn.append(med_fn); q3_fn.append(Q3_fn)
    
    return (q1, median, q3, label, 'Megalodon', accuracy_beta, \
        accuracy_02, accuracy_01, accuracy_005, accuracy_001), (q1_fp, median_fp, q3_fp, label, 'Megalodon'), \
            (q1_fn, median_fn, q3_fn, label, 'Megalodon')


def plot_comparison(deepmp, deepsignal, nanopolish, guppy, megalodon, output):
    # sns.set_theme(style="darkgrid", palette = "Set2")
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')

    custom_lines = []
    for pred in [deepmp, megalodon, deepsignal, nanopolish, guppy]:
        custom_lines.append(
            plt.plot([],[], marker="o", ms=8, ls="", mec='black', 
            mew=0.5, color=palette_dict[pred[4]], label=pred[4])[0] 
        )

        if pred[4] == 'DeepMP':
            pos = -3
        elif pred[4] == 'Megalodon':
            pos = -1.5
        elif pred[4] == 'Nanopolish':
            pos = 1.5
        elif pred[4] == 'Guppy':
            pos = 3
        else:
            pos = 0
        # import pdb;pdb.set_trace()
        yerr = np.array([[np.asarray(pred[1]) - np.asarray(pred[0])], [np.asarray(pred[2]) - np.asarray(pred[1])]])
        yerr = yerr.reshape(2, yerr.shape[2])

        plt.errorbar(
            np.asarray(pred[3]).astype(np.int) + pos, np.asarray(pred[1]), yerr=yerr, 
            marker='o', mfc=palette_dict[pred[4]], mec='black', ms=8, mew=0.5, 
            ecolor='black', capsize=1, capthick=0.5,  fmt=' ', elinewidth=0.8
        )

    ax.set_xlabel("Methylation percentage", fontsize=12)
    ax.set_ylabel("1 - (Inferred Frequency - True Frequency)", fontsize=12)
    plt.xticks(rotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    plt.tight_layout()
    out_dir = os.path.join(output, 'methylation_frequency.pdf')
    plt.savefig(out_dir)
    plt.close()


def plot_fp_fn(deepmp, deepsignal, nanopolish, guppy, megalodon, output, label):
    # sns.set_theme(style="darkgrid", palette = "Set2")
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')

    custom_lines = []
    for pred in [deepmp, deepsignal, nanopolish, guppy, megalodon]:
        custom_lines.append(
            plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
            mew=1, color=palette_dict[pred[4]], label=pred[4])[0] 
        )

        if pred[4] == 'DeepMP':
            pos = -3
        elif pred[4] == 'Megalodon':
            pos = -1.5
        elif pred[4] == 'Nanopolish':
            pos = 1.5
        elif pred[4] == 'Guppy':
            pos = 3
        else:
            pos = 0

        y_axis_labels = pred[3].copy()
        if '0' in y_axis_labels and label == 'Negative': 
            y_axis_labels.remove('0')

        elif '100' in y_axis_labels and label == 'Positive':
            y_axis_labels.remove('100')

        yerr = np.array([[np.asarray(pred[1]) - np.asarray(pred[0])], [np.asarray(pred[2]) - np.asarray(pred[1])]])
        yerr = yerr.reshape(2, yerr.shape[2])

        plt.errorbar(
            np.asarray(y_axis_labels).astype(np.int) + pos, np.asarray(pred[1]), yerr=yerr, 
            marker='o', mfc=palette_dict[pred[4]], mec='black', ms=8, mew=0.5, 
            ecolor='black', capsize=1, capthick=0.5,  fmt=' ', elinewidth=0.8
        )

    ax.set_xlabel("Methylation percentage", fontsize=12)
    if label == 'Positive':
        ax.set_ylabel("False Positive Rate", fontsize=12)
    else:
        ax.set_ylabel("False Negative Rate", fontsize=12)

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
    out_dir = os.path.join(output, 'False_{}_Rate.pdf'.format(label))
    plt.savefig(out_dir)
    plt.close()


def plot_pos_accuracy(deepmp, deepsignal, nanopolish, guppy, megalodon, output):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')

    custom_lines = []
    for pred in [deepmp, deepsignal, nanopolish, guppy, megalodon]:
        custom_lines.append(
            plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
            mew=1, color=palette_dict[pred[4]], label=pred[4])[0] 
        )
        import pdb;pdb.set_trace()
        if pred[4] == 'DeepMP':
            pos = -3
        elif pred[4] == 'Megalodon':
            pos = -1.5
        elif pred[4] == 'Nanopolish':
            pos = 1.5
        elif pred[4] == 'Guppy':
            pos = 3
        else:
            pos = 0

        if pred[4] == 'DeepMPsafas':
            plt.plot(
                np.asarray(pred[3]).astype(np.int)[:-1] + pos, pred[5][:-1], 
                marker='o', mfc=palette_dict[pred[4]], mec='black', ms=7, mew=1, 
                c=palette_dict[pred[4]]
            )
        # elif pred[4] == 'DeepSignal':
        #     pos = 3
        #     plt.plot(
        #         np.asarray(pred[3]).astype(np.int)[1:] + pos, pred[6][1:], 
        #         marker='o', mfc=palette_dict[pred[4]], mec='black', ms=7, mew=1, 
        #         c=palette_dict[pred[4]]
        #     )
        else:
            # import pdb; pdb.set_trace()
            plt.plot(
                np.asarray(pred[3]).astype(np.int)[:-1] + pos, pred[6][:-1], 
                marker='o', mfc=palette_dict[pred[4]], mec='black', ms=7, mew=1, 
                c=palette_dict[pred[4]]
            )

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


def plot_pos_accuracy_beta(deepmp, deepsignal, nanopolish, guppy, megalodon, output):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')

    custom_lines = []
    for pred in [deepmp, deepsignal, nanopolish, guppy, megalodon]:
        custom_lines.append(
            plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
            mew=1, color=palette_dict[pred[4]], label=pred[4])[0] 
        )

        if pred[4] == 'DeepMP':
            pos = -3
        elif pred[4] == 'Megalodon':
            pos = -1.5
        elif pred[4] == 'Nanopolish':
            pos = 1.5
        elif pred[4] == 'Guppy':
            pos = 3
        else:
            pos = 0

        # if pred[4] == 'DeepMP':
        #     pos = -3
        plt.plot(
            np.asarray(pred[3]).astype(np.int)[:-1] + pos, pred[5][:-1], 
            marker='o', mfc=palette_dict[pred[4]], mec='black', ms=7, mew=1, 
            c=palette_dict[pred[4]]
        )
        # elif pred[4] == 'DeepSignal':
        #     pos = 3
        #     plt.plot(
        #         np.asarray(pred[3]).astype(np.int)[1:] + pos, pred[5][1:], 
        #         marker='o', mfc=palette_dict[pred[4]], mec='black', ms=7, mew=1, 
        #         c=palette_dict[pred[4]]
        #     )
        # else:
        #     pos = 0
        #     plt.plot(
        #         np.asarray(pred[3]).astype(np.int)[1:] + pos, pred[5][1:], 
        #         marker='o', mfc=palette_dict[pred[4]], mec='black', ms=7, mew=1, 
        #         c=palette_dict[pred[4]]
        #     )

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

    labels = ['Beta Model', '10{} Threshold'.format('%'), '5{} Threshold'.format('%'), '1{} Threshold'.format('%')]
    positions = [0.2, -0.2, -0.05, 0.05]
    custom_lines = []
    for pred in [deepmp]:
        

        if pred[4] == 'DeepMP':
            
            counter = 1
            pos = -0.1
            for el in pred[5:9]:

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
                    marker='o', mfc=palette_dict[pred[4]], mec='black', ms=7, mew=1, 
                    c=palette_dict[pred[4]]
                )
        else:
            pos = 0
            for el in pred[7:10]:
                plt.plot(
                    np.asarray(pred[3]).astype(np.int) + pos, el, 
                    marker='o', mfc=palette_dict[pred[4]], mec='black', ms=7, mew=1, 
                    c=palette_dict[pred[4]]
                )

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
            mew=0, color=palette_dict[pred[4]], label=pred[4])[0] 
        )

        alg_df = pd.DataFrame(
            [[pred[4], pred[5][0], 'Beta Model'],[pred[4], pred[7][0], '10% Threshold'], \
            [pred[4], pred[8][0], '5% Threshold'], [pred[4], pred[9][0], '1% Threshold']], \
            columns=['Model', 'Accuracy', 'Threshold']
        )
        df = pd.concat([df, alg_df])


    sns.barplot(x="Threshold", y="Accuracy", hue="Model", data=df, 
        palette=['#08519c', '#f03b20','#238443'])

    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel("Position Accuracy", fontsize=12)
    plt.xticks(rotation=0, fontsize=9)

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


def clean_preds(preds):
    import ast; cleaned = []; 
    for el in preds: 
        try:
            cleaned.append(ast.literal_eval(el))
        except: 
            cleaned.append(el)
    return cleaned


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
    '-pdn', '--predictions_nanopolish',
    help='Folder containing the predictions on test from nanopolish'
)
@click.option(
    '-pdg', '--predictions_guppy',
    help='Folder containing the predictions on test from guppy'
)
@click.option(
    '-pdme', '--predictions_megalodon',
    help='Folder containing the predictions on test from megalodon'
)
@click.option(
    '-az', '--around_zero', is_flag=True, default=False, 
    help='whether the inputs are around 0 percentages'
)
@click.option(
    '-o', '--output', default='', help='output folder'
)
def main(predictions_deepmp, predictions_deepsignal, predictions_deepmod, 
    predictions_nanopolish, predictions_guppy, predictions_megalodon, 
    around_zero, output):
    ids = get_ids(predictions_deepmp)

    preds_deepmp, deepmp_fp, deepmp_fn = extract_preds_deepmp(
        predictions_deepmp
    )
    preds_deepsignal, deepsignal_fp, deepsignal_fn = extract_preds_deepsignal(
        predictions_deepsignal, ids
    )
    # preds_deepmod, deepmod_fp, deepmod_fn = extract_preds_deepmod(
    #     predictions_deepmod, output
    # )
    
    preds_nanopolish, nanopolish_fp, nanopolish_fn = extract_preds_nanopolish(
        predictions_nanopolish, ids
    )
    preds_guppy, guppy_fp, guppy_fn = extract_preds_guppy(
        predictions_guppy, ids
    )
    preds_megalodon, megalodon_fp, megalodon_fn = extract_preds_megalodon(
        predictions_megalodon, ids
    )

    import pdb;pdb.set_trace()
    #TODO remove 
    # preds_deepmp = clean_preds(pd.read_csv(os.path.join(output, 'preds_deepmp.tsv'), sep='\t', header=None).values[0])
    # preds_deepsignal = clean_preds(pd.read_csv(os.path.join(output, 'preds_deepsignal.tsv'), sep='\t', header=None).values[0])
    # preds_guppy = clean_preds(pd.read_csv(os.path.join(output, 'preds_guppy.tsv'), sep='\t', header=None).values[0])
    # preds_nanopolish = clean_preds(pd.read_csv(os.path.join(output, 'preds_nanopolish.tsv'), sep='\t', header=None).values[0])
    # preds_megalodon = clean_preds(pd.read_csv(os.path.join(output, 'preds_megalodon.tsv'), sep='\t', header=None).values[0])

    # deepmp_fp = clean_preds(pd.read_csv(os.path.join(output, 'deepmp_fp.tsv'), sep='\t', header=None).values[0])
    # deepsignal_fp = clean_preds(pd.read_csv(os.path.join(output, 'deepsignal_fp.tsv'), sep='\t', header=None).values[0])
    # guppy_fp = clean_preds(pd.read_csv(os.path.join(output, 'guppy_fp.tsv'), sep='\t', header=None).values[0])
    # nanopolish_fp = clean_preds(pd.read_csv(os.path.join(output, 'nanopolish_fp.tsv'), sep='\t', header=None).values[0])
    # megalodon_fp = clean_preds(pd.read_csv(os.path.join(output, 'megalodon_fp.tsv'), sep='\t', header=None).values[0])

    # deepmp_fn = clean_preds(pd.read_csv(os.path.join(output, 'deepmp_fn.tsv'), sep='\t', header=None).values[0])
    # deepsignal_fn = clean_preds(pd.read_csv(os.path.join(output, 'deepsignal_fn.tsv'), sep='\t', header=None).values[0])
    # guppy_fn = clean_preds(pd.read_csv(os.path.join(output, 'guppy_fn.tsv'), sep='\t', header=None).values[0])
    # nanopolish_fn = clean_preds(pd.read_csv(os.path.join(output, 'nanopolish_fn.tsv'), sep='\t', header=None).values[0])
    # megalodon_fn = clean_preds(pd.read_csv(os.path.join(output, 'megalodon_fn.tsv'), sep='\t', header=None).values[0])
    import pdb;pdb.set_trace()
    if around_zero:
        plot_pos_barplot_around_0(
            preds_deepmp, preds_deepsignal, preds_nanopolish, 
            preds_guppy, preds_megalodon, output
        )
        # plot_pos_accuracy_around_0(preds_deepmp, preds_deepsignal, preds_nanopolish, preds_guppy, preds_megalodon, output) 
    
    else:
        plot_comparison(
            preds_deepmp, preds_deepsignal, preds_nanopolish, 
            preds_guppy, preds_megalodon, output)
        # import pdb;pdb.set_trace()
        plot_pos_accuracy(
            preds_deepmp, preds_deepsignal, preds_nanopolish, 
            preds_guppy, preds_megalodon, output)
        plot_pos_accuracy_beta(
            preds_deepmp, preds_deepsignal, preds_nanopolish, 
            preds_guppy, preds_megalodon, output)
        plot_fp_fn(
            deepmp_fp, deepsignal_fp, nanopolish_fp, guppy_fp, 
            megalodon_fp, output, 'Positive'
        )
        plot_fp_fn(
            deepmp_fn, deepsignal_fn, nanopolish_fn, guppy_fn, 
            megalodon_fn, output, 'Negative'
        )


if __name__ == '__main__':
    main()