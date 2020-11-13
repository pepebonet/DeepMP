#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import gamma
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support

import deepmp.utils as ut
import deepmp.plots as pl
import deepmp.preprocess as pr

read1_pos0 = 0.0001  
read0_pos1 = 0.4 
fp = 0.001  
fn = 0.001  
beta_a = 1
beta_b = 10

def test_single_read(data, model_file, labels, score_av='binary'):
    model = load_model(model_file)
    
    pred =  model.predict(data).flatten()
    inferred = np.zeros(len(pred), dtype=int)
    inferred[np.argwhere(pred >= 0.5)] = 1

    #TODO remove
    precision, recall, f_score, _ = precision_recall_fscore_support(
        labels, inferred, average=score_av
    )
    print(precision, recall, f_score)

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
        
        #TODO remove
        if len(df.methyl_label.unique()) == 2:
            meth_label.append(1)
        else:
            meth_label.append(df.methyl_label.unique()[0])

    return pred_label, meth_label


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
    import pdb;pdb.set_trace()
    prob_beta_mod.append(prob_pos_1 / (prob_pos_0 + prob_pos_1))
    prob_beta_unmod.append(prob_pos_0 / (prob_pos_0 + prob_pos_1))

    if prob_mod[-1] >= prob_unmod[-1]:
        pred_beta.append(1)
    else:
        pred_beta.append(0)

    return pred_beta, prob_beta_mod, prob_beta_unmod


def do_per_position_analysis(df, pred_vec, inferred_vec, output, pred_type):
    df['id'] = df['chrom'] + '_' + df['pos'].astype(str)
    df['pred_prob'] = pred_vec
    df['inferred_label'] = inferred_vec
    cov = []; pred_min_max = []; pred_005 = []; pred_01 = []; pred_02 = []
    pred_03 = []; pred_04 = []; meth_label = []; ids = []; pred_posterior = []
    prob_mod = []; prob_unmod = []; pred_beta = []; prob_beta_mod = []
    prob_beta_unmod = []

    for i, j in df.groupby('id'):
        pred_min_max, pred_005, pred_01, pred_02, pred_03, pred_04, meth_label = pred_site_all(
            j, pred_min_max, pred_005, pred_01, pred_02, pred_03, pred_04, meth_label
        )
        pred_posterior, prob_mod, prob_unmod = pred_stats(
            j['inferred_label'].values, pred_posterior, prob_mod, prob_unmod
        )

        pred_beta, prob_beta_mod, prob_beta_unmod = beta_stats(
            j['inferred_label'].values, pred_beta, prob_beta_mod, prob_beta_unmod
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
    preds['meth_label'] = meth_label 

    return preds


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
        pred, inferred = test_single_read([data_seq, data_err], trained_model, labels)
    # ut.save_probs_user(pred, inferred, output)

    ## position-based calling
    # TODO store position info in test file
    if pos_based:
        if 'data_id' in locals():
            test = build_test_df(data_id)
            #TODO DELETE
            test['methyl_label'] = labels

        #TODO output proper df with all the information. put columns at different thresholds as well as the min max for testing
        all_preds = do_per_position_analysis(test, pred, inferred, output, pred_type)
        

        uu = precision_recall_fscore_support(all_preds['meth_label'], all_preds['pred_005'], average='binary')
        xx = precision_recall_fscore_support(all_preds['meth_label'], all_preds['pred_01'], average='binary')
        yy = precision_recall_fscore_support(all_preds['meth_label'], all_preds['pred_02'], average='binary')
        zz = precision_recall_fscore_support(all_preds['meth_label'], all_preds['pred_03'], average='binary')
        ww = precision_recall_fscore_support(all_preds['meth_label'], all_preds['pred_04'], average='binary')
        vv = precision_recall_fscore_support(all_preds['meth_label'], all_preds['pred_min_max'], average='binary')
        pp = precision_recall_fscore_support(all_preds['meth_label'], all_preds['pred_posterior'], average='binary')
        
        # print(xx,yy,zz,ww,vv,pp)
        import pdb;pdb.set_trace()

