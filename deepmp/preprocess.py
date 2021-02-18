#!/usr/bin/env python3

import os
import functools
import subprocess
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.model_selection import train_test_split

import deepmp.utils as ut
import deepmp.merge_h5s as mh5 


names_all = ['chrom', 'pos', 'strand', 'pos_in_strand', 'readname',
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_diff', 'qual', 'mis', 'ins', 'del', 'methyl_label']

names_seq = ['chrom', 'pos', 'strand', 'pos_in_strand', 'readname',
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_diff', 'methyl_label']

names_err =['readname', 'pos', 'chrom', 'kmer', 'qual', 'mis', 'ins', 
            'del', 'methyl_label']


# ------------------------------------------------------------------------------
# PREPROCESS INCUDING TRAIN-TEST-VAL SPLITS
# ------------------------------------------------------------------------------

def get_training_test_val(df):
    train, test = train_test_split(df, test_size=0.05, random_state=0)
    train, val = train_test_split(train, test_size=test.shape[0], random_state=0)
    return [(train, 'train'), (test, 'test'), (val, 'val')]


def get_training_test_val_chr(df):
    test = df[df['chrom'] == 'chr1']
    df_red = df[df['chrom'] != 'chr1']
    train, val = train_test_split(df_red, test_size=0.05, random_state=0)
    return [(train, 'train'), (test, 'test'), (val, 'val')]


def get_training_test_val_pos(df):
    test = df[(df['pos_in_strand'] >= 1000000) & (df['pos_in_strand'] <= 2000000)]
    df_red = pd.concat(
        [df[df['pos_in_strand'] < 1000000], df[df['pos_in_strand'] > 2000000]]
    )
    train, val = train_test_split(df_red, test_size=0.05, random_state=0)
    return [(train, 'train'), (test, 'test'), (val, 'val')]


def save_tsv(df, output, file, mode='w'):
    file_name = os.path.join(output, '{}.tsv'.format(file))
    if mode == 'a':
        df.to_csv(file_name, sep='\t', index=None, mode=mode, header=None)
    else:
        df.to_csv(file_name, sep='\t', index=None, mode=mode)


def get_positions_only(df, positions):
    df = pd.merge(
        df, positions, right_on=['chr', 'start', 'strand'], 
        left_on=['chrom', 'pos', 'strand']
    )

    label = np.zeros(len(df), dtype=int)
    label[np.argwhere(df['status'].values == 'mod')] = 1

    df = df[df.columns[:19]]
    df['methyl_label'] = label

    return df


def split_sets_files(file, tmp_folder, counter, tsv_flag, output, 
    tmps, split_type, positions):
    df = pd.read_csv(os.path.join(tmp_folder, file), sep='\t', names=names_all)

    if isinstance(positions, pd.DataFrame):
        df = get_positions_only(df, positions)

    if split_type == 'read':
        data = get_training_test_val(df)

    elif split_type == 'chr':
        data = get_training_test_val_chr(df)

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
        if el[0].shape[0] > 0:
            ut.preprocess_combined(el[0], tmps, el[1], file)


def do_combined_preprocess(features, output, tsv_flag, cpus, split_type, positions):

    tmp_folder = os.path.join(os.path.dirname(features), 'tmp_all/')
    tmp_train = os.path.join(os.path.dirname(features), 'train/')
    tmp_test = os.path.join(os.path.dirname(features), 'test/')
    tmp_val = os.path.join(os.path.dirname(features), 'val/')

    print('Splitting original file...')
    os.mkdir(tmp_folder); 
    os.mkdir(tmp_train); os.mkdir(tmp_test); os.mkdir(tmp_val)
    cmd = 'split -l {} {} {}'.format(20000, features, tmp_folder)
    subprocess.call(cmd, shell=True)
    
    if positions:
        print('Getting position file...')
        positions = pd.read_csv(positions, sep='\t')

    print('Extracting features to h5 and tsv files...')
    counter = 0
    
    f = functools.partial(split_sets_files, tmp_folder=tmp_folder, \
            counter=counter, tsv_flag=tsv_flag, output=output, \
                tmps=os.path.dirname(features), split_type=split_type, \
                    positions=positions)
    with Pool(cpus) as p:
        for i, rval in enumerate(p.imap_unordered(f, os.listdir(tmp_folder))):
            counter += 1
    
    print('Concatenating features into h5s...')
    mh5.get_set(tmp_test, output, 'test')
    mh5.get_set(tmp_val, output, 'val')
    mh5.get_set(tmp_train, output, 'train')

    print('Removing tmp folders and done')
    subprocess.call('rm -r {}'.format(tmp_folder), shell=True)
    subprocess.call('rm -r {}'.format(tmp_train), shell=True)
    subprocess.call('rm -r {}'.format(tmp_test), shell=True)
    subprocess.call('rm -r {}'.format(tmp_val), shell=True)


# ------------------------------------------------------------------------------
# NO SPLIT PREPROCESS
# ------------------------------------------------------------------------------


def split_sets_files_single(file, tmp_folder, tmps, feature_type):

    if feature_type == 'combined':
        df = pd.read_csv(os.path.join(tmp_folder, file), sep='\t', names=names_all)
        if df.shape[0] != 0:
            ut.preprocess_combined(df, tmps, 'test', file)
    elif feature_type == 'seq':
        df = pd.read_csv(os.path.join(tmp_folder, file), sep='\t', names=names_seq)
        if df.shape[0] != 0:
            ut.preprocess_sequence(df, tmps, 'test', file)
    else:
        df = pd.read_csv(os.path.join(tmp_folder, file), sep='\t', names=names_err)
        if df.shape[0] != 0:
            ut.preprocess_errors(df, tmps, 'test', file)


def no_split_preprocess(features, output, cpus, feature_type):

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
        tmps=os.path.dirname(features), feature_type=feature_type)
    with Pool(cpus) as p:
        for i, rval in enumerate(p.imap_unordered(f, os.listdir(tmp_folder))):
            counter += 1
    
    print('Concatenating features into h5s...')
    mh5.get_set(tmp_test, output, 'test')

    print('Removing tmp folders and done')
    subprocess.call('rm -r {}'.format(tmp_folder), shell=True)