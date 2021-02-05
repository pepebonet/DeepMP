#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import gamma
from tensorflow.keras.models import load_model

import deepmp.utils as ut
import deepmp.preprocess as pr

epsilon = 0.05
gamma_val = 0.8

#E. coli
beta_a = 1
beta_b = 22
beta_c = 14.5

# Human
# beta_a = 1
# beta_b = 6.5
# beta_c = 10.43

# ------------------------------------------------------------------------------
# READ PREDICTION FUNCTIONS
# ------------------------------------------------------------------------------

def do_read_calling(model_type, test_file, trained_model, kmer, err_feat):
    if model_type == 'seq':
        data_seq, labels, data_id = ut.get_data_sequence(
            test_file, kmer, err_feat, get_id=True
        )
        pred, inferred = test_single_read(data_seq, trained_model)

    elif model_type == 'err':
        data_err, labels, data_id = ut.get_data_errors(test_file, kmer, get_id=True)
        pred, inferred = test_single_read(data_err, trained_model)

    elif model_type == 'joint':
        data_seq, data_err, labels, data_id = ut.get_data_jm(
            test_file, kmer, get_id=True
        )
        pred, inferred = test_single_read([data_seq, data_err], trained_model)
    
    return build_test_df(data_id, pred, inferred, model_type)


def test_single_read(data, model_file):
    model = load_model(model_file)

    pred =  model.predict(data).flatten()
    inferred = np.zeros(len(pred), dtype=int)
    inferred[np.argwhere(pred >= 0.5)] = 1

    return pred, inferred



def build_test_df(data, pred_vec, inferred_vec, model_type):
    df = pd.DataFrame()
    df['chrom'] = data[0].astype(str)
    df['pos'] = data[2]

    if model_type != 'err':
        df['strand'] = data[3].astype(str)
        df['pos_in_strand'] = data[4]
        df['readname'] = data[1].astype(str)
    
    df['pred_prob'] = pred_vec
    df['inferred_label'] = inferred_vec

    return df


# ------------------------------------------------------------------------------
# POSITION CALLING FUNCTIONS
# ------------------------------------------------------------------------------

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
    df['id'] = df['chrom'] + '_' + df['pos'].astype(str)

    cov, meth_label, ids, pred_beta = [], [], [], []
    prob_beta_mod, prob_beta_unmod, chromosome, pos = [], [], [], []

    for i, j in df.groupby('id'):
        pred_beta, prob_beta_mod, prob_beta_unmod = beta_stats(
            j['pred_prob'].values, pred_beta, prob_beta_mod, prob_beta_unmod
        )

        cov.append(len(j)); ids.append(i)
        chromosome.append(i.split('_')[0])
        pos.append(i.split('_')[1])

    preds = pd.DataFrame()
    preds['chrom'] = chromosome
    preds['pos'] = pos
    preds['id'] = ids
    preds['cov'] = cov  
    preds['pred_beta'] = pred_beta
    preds['prob_beta_mod'] = prob_beta_mod
    preds['prob_beta_unmod'] = prob_beta_unmod 

    return preds


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def call_mods_user(model_type, test_file, trained_model, kmer, output,
    err_features=False, pos_based=False):

    ## Raise exception for other file formats that are not .h5
    if test_file.rsplit('.')[-1] != 'h5':
        raise Exception('Use .h5 format instead. DeepMP preprocess will get it done')
    
    test = do_read_calling(
        model_type, test_file, trained_model, kmer, err_features
    )
    test.to_csv(os.path.join(
        output, 'read_predictions_DeepMP.tsv'), sep='\t', index=None
    )
    
    ## position-based calling and store
    if pos_based:
        import pdb;pdb.set_trace()
        all_preds = do_per_position_analysis(test)

        all_preds.to_csv(os.path.join(
            output, 'position_calling_DeepMP.tsv'), sep='\t', index=None
        )
        import pdb;pdb.set_trace()

