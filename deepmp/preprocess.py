#!/usr/bin/env python3

import os
import h5py
import functools
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from multiprocessing import Pool
from deepmp.utils import kmer2code
from collections import OrderedDict, Counter
from sklearn.model_selection import train_test_split

import deepmp.utils as ut
import deepmp.merge_h5s as mh5 


names_all=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname',
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_skew', 'signal_kurt', 'signal_diff', 'signal_lens',
            'cent_signals', 'qual', 'mis', 'ins', 'del', 'methyl_label', 'flag', 
            'cent_min2', 'cent_min1', 'cent', 'cent_plus1', 'cent_plus2']


def get_data(treated, untreated, names='', nopos=False):
    if names:
        treat = pd.read_csv(treated, delimiter = "\t", names=names)
        untreat = pd.read_csv(untreated, delimiter = "\t", names=names)
    else:
        if nopos:
            treat = pd.read_csv(treated, sep=',').drop(columns=['pos'])
            untreat = pd.read_csv(untreated, sep=',').drop(columns=['pos'])
        else:
            treat = pd.read_csv(treated, sep=',')
            untreat = pd.read_csv(untreated, sep=',')
    return treat, untreat


def get_training_test_val(df):
    train, test = train_test_split(df, test_size=0.05, random_state=0)
    train, val = train_test_split(train, test_size=0.05, random_state=0)
    return [(train, 'train'), (test, 'test'), (val, 'val')]


def get_training_test_val_pos(df):
    test = df[(df['pos_in_strand'] >= 1000000) & (df['pos_in_strand'] <= 2000000)]
    df_red = pd.concat([df[df['pos_in_strand'] < 1000000], df[df['pos_in_strand'] > 2000000]])
    train, val = train_test_split(df_red, test_size=0.05, random_state=0)
    return [(train, 'train'), (test, 'test'), (val, 'val')]


def preprocess_error(df, feat, output, file):
    X = df[df.columns[:-1]].values
    Y = df[df.columns[-1]].values
    kmer = df['#Kmer'].apply(kmer2code)

    file_name = os.path.join(output, '{}_err.h5'.format(file))

    with h5py.File(file_name, 'a') as hf:
        hf.create_dataset("kmer",  data=np.stack(kmer))
        hf.create_dataset("err_X", data=X.reshape(X.shape[0], feat, 1))
        hf.create_dataset("err_Y", data=Y)

    return None


def do_single_preprocess(feature_type, sequence_treated, sequence_untreated,
    error_treated, error_untreated, output, num_err_feat):
    if feature_type == 'err':
        treat, untreat = get_data(err_treated, err_untreated,
            names=['read_name', 'pos', 'chr', 'k_mer', 'qual', 'mis', 'ins', 'del', 'methyl_label'])
        for el in data:
            ut.preprocess_err_read(el[0], num_err_feat, output, el[1])

    else:
        treat, untreat = get_data(sequence_treated, sequence_untreated,
            names=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname',
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_skew', 'signal_kurt', 'signal_diff',
            'signal_lens', 'methyl_label', 'flag'])
        data = get_training_test_val(pd.concat([treat, untreat]))
        for el in data:
            ut.preprocess_sequence(el[0], output, el[1])


def balanced_set(df):
    label_counts = Counter(df['methyl_label'])
    if len(labels_counts) == 2:
        min_label = min(label_counts, key=label_counts.get)
        zeros = df[df['methyl_label'] == 0]
        ones = df[df['methyl_label'] == 1]

        if min_label == 0:
            ones = ones.sample(label_counts[0])
        else:
            zeros = zeros.sample(label_counts[1])
        
        return pd.concat([ones, zeros])
    
    else:
        return df


def save_tsv(df, output, file, mode='w'):
    file_name = os.path.join(output, '{}.tsv'.format(file))
    if mode == 'a':
        df.to_csv(file_name, sep='\t', index=None, mode=mode, header=None)
    else:
        df.to_csv(file_name, sep='\t', index=None, mode=mode)


def split_sets_files(file, tmp_folder, counter, tsv_flag, output, tmps, split_type):
    df = pd.read_csv(os.path.join(tmp_folder, file), sep='\t', names=names_all)
    # df = balanced_set(df)

    if split_type == 'read':
        data = get_training_test_val(df)
    else:
        data = get_training_test_val_pos(df)
    
    if tsv_flag:
        for el in data:
            if counter == 0:
                mode = 'w'
            else:
                mode = 'a'
            save_tsv(el[0], output, el[1], 'a')
    for el in data:
        ut.preprocess_combined(el[0], tmps, el[1], file)


def split_sets_files_single(file, tmp_folder, counter, tsv_flag, output, tmps, split_type):
    df = pd.read_csv(os.path.join(tmp_folder, file), sep='\t', names=names_all)

    if split_type == 'read':
        data = [(df, 'test')]
    else:
        test = df[(df['pos_in_strand'] >= 1000000) & (df['pos_in_strand'] <= 2000000)]
        data = [(test, 'test')]
    
    if data[0][0].shape[0] != 0:
        if tsv_flag:
            for el in data:
                if counter == 0:
                    mode = 'w'
                else:
                    mode = 'a'
                save_tsv(el[0], output, el[1], 'a')
        for el in data:
            ut.preprocess_combined(el[0], tmps, el[1], file)


def do_combined_preprocess(features, output, tsv_flag, cpus, split_type):

    tmp_folder = os.path.join(os.path.dirname(features), 'tmp_shuffled/')
    tmp_train = os.path.join(os.path.dirname(features), 'train/')
    tmp_test = os.path.join(os.path.dirname(features), 'test/')
    tmp_val = os.path.join(os.path.dirname(features), 'val/')

    print('Splitting original file...')
    # os.mkdir(tmp_folder); 
    # os.mkdir(tmp_train); os.mkdir(tmp_test); os.mkdir(tmp_val)
    # cmd = 'split -l {} {} {}'.format(20000, features, tmp_folder)
    # subprocess.call(cmd, shell=True)
    
    print('Extracting features to h5 and tsv files...')
    counter = 0
    
    f = functools.partial(split_sets_files, tmp_folder=tmp_folder, \
            counter=counter, tsv_flag=tsv_flag, output=output, \
                tmps=os.path.dirname(features), split_type=split_type)
    with Pool(cpus) as p:
        for i, rval in enumerate(p.imap_unordered(f, os.listdir(tmp_folder))):
            counter += 1

    print('Concatenating features into h5s...')
    mh5.get_set(tmp_test, output, 'test')
    mh5.get_set(tmp_val, output, 'val')
    mh5.get_set(tmp_train, output, 'train')

    print('Removing tmp folders and done')
    # subprocess.call('rm -r {}'.format(tmp_folder), shell=True)
    subprocess.call('rm -r {}'.format(tmp_train), shell=True)
    subprocess.call('rm -r {}'.format(tmp_test), shell=True)
    subprocess.call('rm -r {}'.format(tmp_val), shell=True)


def no_split_combined_preprocess(features, output, tsv_flag, cpus, split_type):

    tmp_folder = os.path.join(os.path.dirname(features), 'tmp_all/')
    tmp_test = os.path.join(os.path.dirname(features), 'test/')

    print('Splitting original file...')
    os.mkdir(tmp_folder)
    os.mkdir(tmp_test)
    cmd = 'split -l {} {} {}'.format(20000, features, tmp_folder) 
    subprocess.call(cmd, shell=True)
    
    print('Extracting features to h5 and tsv files...')
    counter = 0

    f = functools.partial(split_sets_files_single, tmp_folder=tmp_folder, \
            counter=counter, tsv_flag=tsv_flag, output=output, \
                tmps=os.path.dirname(features), split_type=split_type)
    with Pool(cpus) as p:
        for i, rval in enumerate(p.imap_unordered(f, os.listdir(tmp_folder))):
            counter += 1
    
    print('Concatenating features into h5s...')
    mh5.get_set(tmp_test, output, 'test')

    print('Removing tmp folders and done')
    # subprocess.call('rm -r {}'.format(tmp_folder), shell=True)
    subprocess.call('rm -r {}'.format(tmp_test), shell=True)