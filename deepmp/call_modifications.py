#!/usr/bin/env python3
import os
import time
import functools
import subprocess
import numpy as np
import pandas as pd
import bottleneck as bn
import tensorflow as tf
import multiprocessing as mp
from scipy.special import gamma
from multiprocessing import Pool
from tensorflow.keras.models import load_model

import deepmp.utils as ut
from deepmp.model import *

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

read_names = ['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 'pred_prob',
       'inferred_label']


queen_size_border = 2000
time_wait = 5

# ------------------------------------------------------------------------------
# READ PREDICTION MULTIPROCESSING
# ------------------------------------------------------------------------------

def _write_predictions_to_file(write_fp, featurestr_q):
    while True:
        if featurestr_q.empty():
            time.sleep(time_wait)
            continue
        features_str = featurestr_q.get()
        try:
            features_str.to_csv(
                write_fp, sep='\t', mode='a', index=None, header=None
            )

        except: 
            if features_str == 'kill':
                break
            else: 
                print(features_str)
                break


def _fill_files_queue(h5s_q, h5s_files, batch_size):
    for i in np.arange(0, len(h5s_files), batch_size):
        h5s_q.put(h5s_files[i:(i+batch_size)])
    return


def do_multiprocessing_main(h5s_q, featurestr_q, errornum_q, model_type, 
    trained_model, kmer, err_feat):
    #Obtain features from every read 
    while not h5s_q.empty():
        try:
            h5s = h5s_q.get()
        except Exception:
            break

        predictions, error_num = do_read_calling_multiprocessing(
            h5s, model_type, trained_model, kmer, err_feat
        )
        # print(predictions, error_num)
        # features_str = []
        # for features in features_list:
        #     features_str.append(_features_to_str(features))

        errornum_q.put(error_num)
        featurestr_q.put(predictions)

        while featurestr_q.qsize() > queen_size_border:
            time.sleep(time_wait)


def do_read_calling_multiprocessing(h5s, model_type, trained_model, kmer, err_feat):
    predictions = pd.DataFrame()

    error = 0
    for h5_fp in h5s:
        try:
            if model_type == 'seq':
                pred, inferred, data_id = seq_read_calling(
                    h5_fp, kmer, err_feat, trained_model, model_type
                )

            elif model_type == 'err':
                pred, inferred, data_id = err_read_calling(
                    h5_fp, kmer, trained_model, model_type
                )

            elif model_type == 'joint':
                pred, inferred, data_id = joint_read_calling(
                    h5_fp, kmer, trained_model, model_type
                )

            test = build_test_df(data_id, pred, inferred, model_type)
            predictions = pd.concat([predictions, test])

        except Exception:
            error += 1
            print(error)
            continue
        
    return predictions, error


# ------------------------------------------------------------------------------
# READ PREDICTION SINGLE
# ------------------------------------------------------------------------------

def do_read_calling(test_file, model_type, trained_model, kmer, err_feat, 
    out_file, tmp_folder='', flag='multi'):
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

    test = build_test_df(data_id, pred, inferred, model_type)
    save_test(test, out_file, tmp_folder, test_file, flag)


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


def save_test(test, out_file, tmp_folder, test_file, flag):
    if flag == 'multi':
        tmp_file = os.path.join(
            tmp_folder, test_file.rsplit('.', 1)[0].rsplit('/', 1)[1] + '.tsv'
        )
        test.to_csv(tmp_file, sep='\t', index=None, header=None)
    else:
        test.to_csv(out_file, sep='\t', index=None, header=None)


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
    meth_freq = []

    for i, j in df.groupby('id'):
        pred_beta, prob_beta_mod, prob_beta_unmod = beta_stats(
            j['pred_prob'].values, pred_beta, prob_beta_mod, prob_beta_unmod
        )

        meth_freq.append(round(j['inferred_label'].sum() / j.shape[0], 5))
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
    preds['meth_freq'] = meth_freq

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
    meth_freq = []

    for i, j in df.groupby('id'):

        pred_threshold = pred_site_threshold(
            j['pred_prob'].values, pred_threshold, threshold
        )

        meth_freq.append(round(j['inferred_label'].sum() / j.shape[0], 5))
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
    preds['meth_freq'] = meth_freq

    return preds


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def do_multiprocessing_reads(test_file, model_type, trained_model, kmer, 
    err_features, reads_output, nproc, output):
    start = time.time()
    h5s_files = [os.path.join(test_file, f) for f in os.listdir(test_file)]

    h5s_q = mp.Queue()
    featurestr_q = mp.Queue()
    errornum_q = mp.Queue()

    _fill_files_queue(h5s_q, h5s_files, 5)
    import pdb;pdb.set_trace()

    print('Getting fread predictions from h5s...')
    #Start process for feature extraction in every core
    featurestr_procs = []
    if nproc > 1:
        nproc -= 1
    for _ in range(nproc):
        p = mp.Process(
            target=do_multiprocessing_main, args=(h5s_q, featurestr_q, 
            errornum_q, model_type, trained_model, kmer, err_features)
        )
        p.daemon = True
        p.start()
        featurestr_procs.append(p)

    print("Writing predictions to file...")
    p_w = mp.Process(
        target=_write_predictions_to_file, args=(reads_output, featurestr_q)
    )
    p_w.daemon = True 
    p_w.start()

    errornum_sum = 0
    while True:
        running = any(p.is_alive() for p in featurestr_procs)
        while not errornum_q.empty():
            errornum_sum += errornum_q.get()
        if not running:
            break

    for p in featurestr_procs:
        p.join() 

    print("finishing the writing process..")
    featurestr_q.put("kill")

    p_w.join()

    print("%d of %d h5 files failed..\n"
          "read predictions costs %.1f seconds.." % (errornum_sum, len(h5s_files),
                                                     time.time() - start))


def do_single_reads(test_file, model_type, trained_model, kmer, 
    err_features, reads_output):
    ## Raise exception for other file formats that are not .h5
    if test_file.rsplit('.')[-1] != 'h5':
        raise Exception('Use .h5 format instead. DeepMP preprocess will get it done')
    
    ## read calling and store
    do_read_calling(
        test_file, model_type, trained_model, kmer, err_features, 
        reads_output, '', 'single'
    )


def do_position_calling(reads_output, use_threshold, threshold, output, model_type):
    test = pd.read_csv(reads_output, sep='\t', names=read_names)
    
    if use_threshold:
        all_preds = do_per_position_theshold(test, threshold)
    
    else:
        all_preds = do_per_position_beta(test)
    
    pos_output = os.path.join(
        output, 'position_calling_{}_DeepMP.tsv'.format(model_type))
    all_preds.to_csv(pos_output, sep='\t', index=None)


def call_mods_user(model_type, test_file, trained_model, kmer, output,
    err_features, pos_based, use_threshold, threshold, cpus):

    reads_output = os.path.join(
            output, 'read_predictions_{}_DeepMP.tsv').format(model_type)

    # read-based calling
    if os.path.isdir(test_file):
        do_multiprocessing_reads(
            test_file, model_type, trained_model, kmer, err_features, 
            reads_output, cpus, output
        )

    else:
        do_single_reads(
            test_file, model_type, trained_model, kmer, err_features,
            reads_output
        )
    
    ## position-based calling
    if pos_based:
        print('Performing position analysis...')

        do_position_calling(
            reads_output, use_threshold, threshold, output, model_type
        )