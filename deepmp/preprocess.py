#!/usr/bin/env python3
import os
import h5py
import check
import click
import functools
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from multiprocessing import Pool
from deepmp.utils import kmer2code
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import hdf5
import msg
from tqdm import tqdm


names_all=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_skew', 'signal_kurt', 'signal_diff', 'signal_lens', 
            'cent_signals', 'qual', 'mis', 'ins', 'del', 'methyl_label', 'flag']


def merge_data(data_list):

    """Merge dictionaries with data.

    Keyword arguments:
    data_list -- the dictionary with data dictionaries
    """

    data = None

    for f in data_list:
        size = check.get_size(data_list[f])
        if not data:
            print("\nThe following datasets were found in %s:\n" % f)
            msg.list_dataset(data_list[f])
            data = data_list[f]
        else:
            print("\nAdding %(n)d entries from %(f)s" % {"n": size, "f": f})
            check.check_keys(data, data_list[f])
            check.check_shapes(data, data_list[f])
            for key in data_list[f]:
                data[key] = np.append(data[key], data_list[f][key], axis=0)

    return data


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


def get_merge_data(errors, sequence):
    try: 
        return pd.merge(sequence, errors, on='pos', how='inner')
    except:
        errors['pos'] = errors['pos'].astype('int64')
        return pd.merge(sequence, errors, on='pos', how='inner')
    

def get_training_test_val(df):
    train, test = train_test_split(df, test_size=0.05, random_state=0)
    train, val = train_test_split(train, test_size=test.shape[0], random_state=0)
    return [(train, 'train'), (test, 'test'), (val, 'val')]


def preprocess_sequence(df, output, file):

    df = df.dropna()
    kmer = df['kmer'].apply(kmer2code)
    base_mean = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_means'].values]
    base_std = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_stds'].values]
    base_median = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_median'].values]
    base_skew = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_skew'].values]
    base_kurt = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_kurt'].values]
    base_diff = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_diff'].values]
    base_signal_len = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_lens'].values]
    label = df['methyl_label']

    file_name = os.path.join(output, '{}_seq.h5'.format(file))

    with h5py.File(file_name, 'a') as hf:
        hf.create_dataset("kmer",  data=np.stack(kmer))
        hf.create_dataset("signal_means",  data=np.stack(base_mean))
        hf.create_dataset("signal_stds",  data=np.stack(base_std))
        hf.create_dataset("signal_median",  data=np.stack(base_median))
        hf.create_dataset("signal_skew",  data=np.stack(base_skew))
        hf.create_dataset("signal_kurt",  data=np.stack(base_kurt))
        hf.create_dataset("signal_diff",  data=np.stack(base_diff))
        hf.create_dataset("signal_lens",  data=np.stack(base_signal_len))
        hf.create_dataset("label",  data=label)

    return None


def preprocess_error(df, feat, output, file):

    X = df[df.columns[:-1]].values
    Y = df[df.columns[-1]].values

    file_name = os.path.join(output, '{}_err.h5'.format(file))

    with h5py.File(file_name, 'a') as hf:
        hf.create_dataset("err_X", data=X.reshape(X.shape[0], feat, 1))
        hf.create_dataset("err_Y", data=Y)

    return None


def preprocess_both(data, err_feat, output, file):

    data = data.dropna()
    data_seq = data[data.columns[:12]]
    data_err = data[data.columns[12:]]

    #sequence
    kmer = data_seq['kmer'].apply(kmer2code)
    base_mean = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in data_seq['signal_means'].values]
    base_std = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in data_seq['signal_stds'].values]
    base_signal_len = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in data_seq['signal_lens'].values]
    label = data_seq['methyl_label']

    #error
    X = data_err[data_err.columns[:-1]].values
    Y = data_err[data_err.columns[-1]].values

    file_name = os.path.join(output, '{}.h5'.format(file))

    with h5py.File(file_name, 'a') as hf:
        hf.create_dataset("kmer",  data=np.stack(kmer))
        hf.create_dataset("signal_means",  data=np.stack(base_mean))
        hf.create_dataset("signal_stds",  data=np.stack(base_std))
        hf.create_dataset("signal_lens",  data=np.stack(base_signal_len))
        hf.create_dataset("label",  data=label)
        hf.create_dataset("err_X", data=X.reshape(X.shape[0], err_feat, 1))
        hf.create_dataset("err_Y", data=Y)

    return None


def do_seq_err_preprocess(sequence_treated, sequence_untreated, 
    error_treated, error_untreated, output, num_err_feat):
    seq_treat, seq_untreat = get_data(sequence_treated, sequence_untreated, 
            names=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_lens', 
            'cent_signals', 'methyl_label'])
    err_treat, err_untreat = get_data(error_treated, error_untreated)
    
    treat = get_merge_data(err_treat, seq_treat)
    untreat = get_merge_data(err_untreat, seq_untreat)

    data = get_training_test_val(pd.concat([treat, untreat]))
    
    for el in data:
        preprocess_both(el[0], num_err_feat, output, el[1])


def do_single_preprocess(feature_type, sequence_treated, sequence_untreated, 
    error_treated, error_untreated, output, num_err_feat):
    if feature_type == 'err':
        treat, untreat = get_data(error_treated, error_untreated, nopos=True)
        data = get_training_test_val(pd.concat([treat, untreat]))
        for el in data:
            preprocess_error(el[0], num_err_feat, output, el[1])

    else: 
        treat, untreat = get_data(sequence_treated, sequence_untreated, 
            names=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_skew', 'signal_kurt', 'signal_diff',  
            'signal_lens', 'methyl_label', 'flag'])
        data = get_training_test_val(pd.concat([treat, untreat]))
        for el in data:
            preprocess_sequence(el[0], output, el[1])


def preprocess_combined(df, output, label_file, file):
    
    kmer = df['kmer'].apply(kmer2code)
    base_mean = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_means'].values]
    base_std = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_stds'].values]
    base_median = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_median'].values]
    base_skew = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_skew'].values]
    base_kurt = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_kurt'].values]
    base_diff = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_diff'].values]
    base_signal_len = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_lens'].values]
    cent_signals = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['cent_signals'].values]
    base_qual = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['qual'].values]
    base_mis = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['mis'].values]
    base_ins = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['ins'].values]
    base_del = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['del'].values]
    label = df['methyl_label']

    file_name = os.path.join(
        output, '{}'.format(label_file), '{}_{}.h5'.format(file, label_file)
    )

    with h5py.File(file_name, 'a') as hf:
        hf.create_dataset("kmer",  data=np.stack(kmer), chunks=True, maxshape=(None,None))
        hf.create_dataset("signal_means",  data=np.stack(base_mean), chunks=True, maxshape=(None,None))
        hf.create_dataset("signal_stds",  data=np.stack(base_std), chunks=True, maxshape=(None,None))
        hf.create_dataset("signal_median",  data=np.stack(base_median), chunks=True, maxshape=(None,None))
        hf.create_dataset("signal_skew",  data=np.stack(base_skew), chunks=True, maxshape=(None,None))
        hf.create_dataset("signal_kurt",  data=np.stack(base_kurt), chunks=True, maxshape=(None,None))
        hf.create_dataset("signal_diff",  data=np.stack(base_diff), chunks=True, maxshape=(None,None))
        hf.create_dataset("signal_lens",  data=np.stack(base_signal_len), chunks=True, maxshape=(None,None))
        hf.create_dataset("signal_central",  data=np.stack(cent_signals), chunks=True, maxshape=(None,None))
        hf.create_dataset('qual',  data=np.stack(base_qual), chunks=True, maxshape=(None,None))
        hf.create_dataset('mis',  data=np.stack(base_mis), chunks=True, maxshape=(None,None))
        hf.create_dataset('ins',  data=np.stack(base_ins), chunks=True, maxshape=(None,None))
        hf.create_dataset('dele',  data=np.stack(base_del), chunks=True, maxshape=(None,None))
        hf.create_dataset('methyl_label',  data=label, chunks=True, maxshape=(None,))
    # else:
    # with h5py.File(file_name, 'a') as hf:
    #     hf["kmer"].resize((hf["kmer"].shape[0] + kmer.shape[0]), axis=0)
    #     hf["kmer"][-kmer.shape[0]:] = np.stack(kmer)
    #     hf["signal_means"].resize((hf["signal_means"].shape[0] + np.stack(base_mean).shape[0]), axis=0)
    #     hf["signal_means"][-np.stack(base_mean).shape[0]:] = np.stack(base_mean)
    #     hf["signal_stds"].resize((hf["signal_stds"].shape[0] + np.stack(base_std).shape[0]), axis=0)
    #     hf["signal_stds"][-np.stack(base_std).shape[0]:] = np.stack(base_std)
    #     hf["signal_median"].resize((hf["signal_median"].shape[0] + np.stack(base_median).shape[0]), axis=0)
    #     hf["signal_median"][-np.stack(base_median).shape[0]:] = np.stack(base_median)
    #     hf["signal_skew"].resize((hf["signal_skew"].shape[0] + np.stack(base_skew).shape[0]), axis=0)
    #     hf["signal_skew"][-np.stack(base_skew).shape[0]:] = np.stack(base_skew)
    #     hf["signal_kurt"].resize((hf["signal_kurt"].shape[0] + np.stack(base_kurt).shape[0]), axis=0)
    #     hf["signal_kurt"][-np.stack(base_kurt).shape[0]:] = np.stack(base_kurt)
    #     hf["signal_diff"].resize((hf["signal_diff"].shape[0] + np.stack(base_diff).shape[0]), axis=0)
    #     hf["signal_diff"][-np.stack(base_diff).shape[0]:] = np.stack(base_diff)
    #     hf["signal_lens"].resize((hf["signal_lens"].shape[0] + np.stack(base_signal_len).shape[0]), axis=0)
    #     hf["signal_lens"][-np.stack(base_signal_len).shape[0]:] = np.stack(base_signal_len)
    #     hf["signal_central"].resize((hf["signal_central"].shape[0] + np.stack(cent_signals).shape[0]), axis=0)
    #     hf["signal_central"][-np.stack(cent_signals).shape[0]:] = np.stack(cent_signals)
    #     hf["qual"].resize((hf["qual"].shape[0] + np.stack(base_qual).shape[0]), axis=0)
    #     hf["qual"][-np.stack(base_qual).shape[0]:] = np.stack(base_qual)
    #     hf["mis"].resize((hf["mis"].shape[0] + np.stack(base_mis).shape[0]), axis=0)
    #     hf["mis"][-np.stack(base_mis).shape[0]:] = np.stack(base_mis)
    #     hf["ins"].resize((hf["ins"].shape[0] + np.stack(base_ins).shape[0]), axis=0)
    #     hf["ins"][-np.stack(base_ins).shape[0]:] = np.stack(base_ins)
    #     hf["dele"].resize((hf["dele"].shape[0] + np.stack(base_del).shape[0]), axis=0)
    #     hf["dele"][-np.stack(base_del).shape[0]:] = np.stack(base_del)
    #     hf["methyl_label"].resize((hf["methyl_label"].shape[0] + label.shape[0]), axis=0)
    #     hf["methyl_label"][-label.shape[0]:] = np.stack(label)
    
    return None


# def initialize_h5(file_name):
#     with h5py.File(file_name, 'a') as hf:
#         # import pdb;pdb.set_trace()
#         hf.create_dataset("kmer", (0,0), chunks=True, maxshape=(None,None))
#         hf.create_dataset("signal_means", (0,0), chunks=True, maxshape=(None,None))
#         hf.create_dataset("signal_stds", (0,0), chunks=True, maxshape=(None,None))
#         hf.create_dataset("signal_median", (0,0), chunks=True, maxshape=(None,None))
#         hf.create_dataset("signal_skew", (0,0), chunks=True, maxshape=(None,None))
#         hf.create_dataset("signal_kurt", (0,0), chunks=True, maxshape=(None,None))
#         hf.create_dataset("signal_diff", (0,0), chunks=True, maxshape=(None,None))
#         hf.create_dataset("signal_lens", (0,0), chunks=True, maxshape=(None,None))
#         hf.create_dataset("signal_central", (0,0), chunks=True, maxshape=(None,None))
#         hf.create_dataset('qual', (0,0), chunks=True, maxshape=(None,None))
#         hf.create_dataset('mis', (0,0), chunks=True, maxshape=(None,None))
#         hf.create_dataset('ins', (0,0), chunks=True, maxshape=(None,None))
#         hf.create_dataset('dele', (0,0), chunks=True, maxshape=(None,None))
#         hf.create_dataset('methyl_label', (0,), chunks=True, maxshape=(None,))



def save_tsv(df, output, file, mode='w'):
    file_name = os.path.join(output, '{}.tsv'.format(file))
    if mode == 'a':
        df.to_csv(file_name, sep='\t', index=None, mode=mode, header=None)
    else:
        df.to_csv(file_name, sep='\t', index=None, mode=mode)


def split_sets_files(file, tmp_folder, counter, tsv_flag, output, tmps):
    df = pd.read_csv(os.path.join(tmp_folder, file), sep='\t', names=names_all)
            
    data = get_training_test_val(df)
    if tsv_flag:
        for el in data:
            if counter == 0:
                mode = 'w'
            else:
                mode = 'a'
            save_tsv(el[0], output, el[1], mode)
    for el in data:
        preprocess_combined(el[0], tmps, el[1], file)


def get_set(folder, output, label):
    filelist = [os.path.join(folder, el) for el in os.listdir(folder)]
    data = OrderedDict()

    for f in filelist:
        data[f] = hdf5.load(f)

    out_file = os.path.join(output, '{}_combined.h5'.format(label))
    hdf5.save(output, merge_data(data))


def do_combined_preprocess(features, output, tsv_flag, mem_efficient, cpus):

    if mem_efficient:
        tmp_folder = os.path.join(os.path.dirname(features), 'tmp1/')
        tmp_train = os.path.join(os.path.dirname(features), 'train/')
        tmp_test = os.path.join(os.path.dirname(features), 'test/')
        tmp_val = os.path.join(os.path.dirname(features), 'val/')

        print('Splitting original file...')
        # os.mkdir(tmp_folder); os.mkdir(tmp_train); os.mkdir(tmp_test); os.mkdir(tmp_val)
        cmd = 'split -l {} {} {}'.format(20000, features, tmp_folder) 
        # subprocess.call(cmd, shell=True)
        
        print('Extracting features to h5 and tsv files...')
        # counter = 0
        # f = functools.partial(split_sets_files, tmp_folder=tmp_folder, \
        #         counter=counter, tsv_flag=tsv_flag, output=output, \
        #             tmps=os.path.dirname(features))
        # with Pool(cpus) as p:
        #     for i, rval in enumerate(p.imap_unordered(f, os.listdir(tmp_folder))):
        #         counter += 1
        
        print('Concatenating features into h5s...')
        get_set(tmp_train, output, 'train')
        get_set(tmp_test, output, 'test')
        get_set(tmp_val, otuput, 'val')
        import pdb;pdb.set_trace()

        print('Removing tmp folders and done')
        subprocess.call('rm -r {}'.format(tmp_folder), shell=True)
        subprocess.call('rm -r {}'.format(tmp_trian), shell=True)
        subprocess.call('rm -r {}'.format(tmp_test), shell=True)
        subprocess.call('rm -r {}'.format(tmp_val), shell=True)

    else: 
        df = pd.read_csv(features, sep='\t', names=names_all)
        data = get_training_test_val(pd.concat([treat, untreat]))

        if tsv_flag:
            for el in data:
                save_tsv(el[0], output, el[1])
        for el in data:
            preprocess_combined(el[0], output, el[1])
        