#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import bottleneck as bn
import tensorflow as tf
from scipy.special import gamma
from tensorflow.keras.models import load_model

import deepmp.utils as ut
from deepmp.model import *
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

EPSILON = np.finfo(np.float64).resolution
log_EPSILON = np.log(EPSILON)


# ------------------------------------------------------------------------------
# READ PREDICTION FUNCTIONS
# ------------------------------------------------------------------------------

def do_read_calling(model_type, test_file, trained_model, kmer, err_feat):
    if model_type == 'seq':
        pred, inferred, data_id = seq_read_calling(
            test_file, kmer, err_feat, trained_model, model_type
        )

    elif model_type == 'err':
        pred, inferred, data_id = err_read_calling(
            test_file, kmer, trained_model, model_type
        )

    elif model_type == 'joint':
        pred, inferred, data_id = joint_read_calling(
            test_file, kmer, trained_model, model_type
        )

    return build_test_df(data_id, pred, inferred, model_type)


def seq_read_calling(test_file, kmer, err_feat, trained_model, model_type):
    data_seq, labels, data_id = ut.get_data_sequence(
            test_file, kmer, err_feat, get_id=True
        )
    pred, inferred = test_single_read(data_seq, trained_model, model_type, kmer)

    return pred, inferred, data_id


def err_read_calling(test_file, kmer, trained_model, model_type):
    data_err, labels, data_id = ut.get_data_errors(test_file, kmer, get_id=True)
    pred, inferred = test_single_read(data_err, trained_model, model_type, kmer)

    return pred, inferred, data_id


def joint_read_calling(test_file, kmer, trained_model, model_type):
    data_seq, data_err, labels, data_id = ut.get_data_jm(
        test_file, kmer, get_id=True
    )
    pred, inferred = test_single_read(
        [data_seq, data_err], trained_model, model_type, kmer
    )
    return pred, inferred, data_id


def test_single_read(data, model_file, model_type, kmer):
    try:
        
        model = load_model(model_file)
    except:
        model = load_model_weights(model_file, model_type, kmer)

    pred =  model.predict(data).flatten()
    inferred = np.zeros(len(pred), dtype=int)
    inferred[np.argwhere(pred >= 0.5)] = 1

    return pred, inferred


def load_model_weights(trained_model, model_type, kmer):
    if model_type == 'seq':
        return load_sequence_weights(trained_model, kmer)

    elif model_type == 'err':
        return load_error_weights(trained_model, kmer)

    else:
        return load_joint_weights(trained_model, kmer)


def load_sequence_weights(trained_model, kmer):
    model = SequenceCNN('conv', 6, 256, 4)
    input_shape = (None, kmer, 9)
    model.compile(loss='binary_crossentropy',
                            optimizer=tf.keras.optimizers.Adam(),
                            metrics=['accuracy'])
    model.build(input_shape)
    model.load_weights(trained_model)

    return model


def load_error_weights(trained_model, kmer):
    model = BCErrorCNN(3, 3, 128, 3)
    model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
    input_shape = (None, kmer, 9)
    model.build(input_shape)
    model.load_weights(trained_model)

    return model


def load_joint_weights(trained_model, kmer):
    model = JointNN()
    model.compile(loss='binary_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00125),
                    metrics=['accuracy'])
    input_shape = ([(None, kmer, 9), (None, kmer, 9)])
    model.build(input_shape)
    model.load_weights(trained_model)

    return model


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

## beta model prediction
def beta_fct(a, b):
        return gamma(a) * gamma(b) / gamma(a + b)


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


def do_per_position_beta(df):
    df['id'] = df['chrom'] + '_' + df['pos'].astype(str) + '_' + df['strand']

    cov, meth_label, ids, pred_beta = [], [], [], []
    prob_beta_mod, prob_beta_unmod, chromosome, pos, strand = [], [], [], [], []

    for i, j in df.groupby('id'):
        pred_beta, prob_beta_mod, prob_beta_unmod = beta_stats(
            j['pred_prob'].values, pred_beta, prob_beta_mod, prob_beta_unmod
        )

        cov.append(len(j)); ids.append(i)
        chromosome.append(i.split('_')[0])
        strand.append(i.split('_')[2])
        pos.append(i.split('_')[1])

    preds = pd.DataFrame()
    preds['chrom'] = chromosome
    preds['pos'] = pos
    preds['strand'] = strand
    preds['id'] = ids
    preds['cov'] = cov  
    preds['pred_beta'] = pred_beta
    preds['prob_beta_mod'] = prob_beta_mod
    preds['prob_beta_unmod'] = prob_beta_unmod 

    return preds

## threshold prediction
def pred_site_threshold(inferred, pred_threshold, threshold):

    if np.sum(inferred) / len(inferred) >= threshold:
        pred_threshold.append(1)
    else:
        pred_threshold.append(0)
    
    return pred_threshold


def do_per_position_theshold(df, threshold):
    df['id'] = df['chrom'] + '_' + df['pos'].astype(str) + '_' + df['strand']

    cov, meth_label, ids, pred_threshold = [], [], [], []
    chromosome, pos, strand = [], [], []

    for i, j in df.groupby('id'):

        pred_threshold = pred_site_threshold(
            j['pred_prob'].values, pred_threshold, threshold
        )

        cov.append(len(j)); ids.append(i)
        strand.append(i.split('_')[2])
        chromosome.append(i.split('_')[0])
        pos.append(i.split('_')[1])

    preds = pd.DataFrame()
    preds['chrom'] = chromosome
    preds['pos'] = pos
    preds['strand'] = strand
    preds['id'] = ids
    preds['cov'] = cov  
    preds['pred_threshold'] = pred_threshold

    return preds


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def call_mods_user(model_type, test_file, trained_model, kmer, output,
    err_features, pos_based, use_threshold, threshold):

    ## Raise exception for other file formats that are not .h5
    if test_file.rsplit('.')[-1] != 'h5':
        raise Exception('Use .h5 format instead. DeepMP preprocess will get it done')
    
    ## read calling and store
    test = do_read_calling(
        model_type, test_file, trained_model, kmer, err_features
    )
    reads_output = os.path.join(
        output, 'read_predictions_{}_DeepMP.tsv').format(model_type)
    test.to_csv(reads_output, sep='\t', index=None)
    
    ## position-based calling and store
    if pos_based:
        
        # test = pd.read_csv(test_file, sep='\t')
        if use_threshold:
            all_preds = do_per_position_theshold(test, threshold)
        
        else:
            all_preds = do_per_position_beta(test)

        pos_output = os.path.join(
            output, 'position_calling_{}_DeepMP.tsv'.format(model_type))
        all_preds.to_csv(pos_output, sep='\t', index=None)


