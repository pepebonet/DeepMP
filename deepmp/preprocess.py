#!/usr/bin/env python3
import os
import h5py
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
from tqdm import tqdm


names_all=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname',
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_skew', 'signal_kurt', 'signal_diff', 'signal_lens',
            'cent_signals', 'qual', 'mis', 'ins', 'del', 'methyl_label', 'flag']


def check_shapes(data1, data2):
    for key in data1.keys():
        if data1[key].shape[1:] != data2[key].shape[1:]:
            raise ValueError("Different shapes for dataset: %s. " % key)


def check_keys(data1, data2):
    if data1.keys() != data2.keys():
        raise ValueError("Files have different datasets.")


def get_size(data):

    sizes = [d.shape[0] for d in data.values()]

    if max(sizes) != min(sizes):
        raise ValueError("Each dataset within a file must have the "
                  "same number of entries!")

    return sizes[0]


def merge_data(data_list):

    data = None

    for f in data_list:
        size = get_size(data_list[f])
        if not data:
            data = data_list[f]
        else:
            check_keys(data, data_list[f])
            check_shapes(data, data_list[f])
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

    #df = df.dropna()
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
        hf.create_dataset("methyl_label",  data=label)

    return None


def preprocess_error(df, feat, output, file):
    import pdb;pdb.set_trace()
    # X = df[df.columns[:-1]].values
    # Y = df[df.columns[-1]].values
    kmer = df['#Kmer'].apply(kmer2code)
    X = df[df.columns[5:-1]].values
    Y = df[df.columns[-1]].values

    file_name = os.path.join(output, '{}_err.h5'.format(file))

    with h5py.File(file_name, 'a') as hf:
        hf.create_dataset("kmer",  data=np.stack(kmer))
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
        treat, untreat = get_data(error_treated, error_untreated, nopos=False)
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
        #output, '{}'.format(label_file), '{}_{}.h5'.format(file, label_file)
        output, '{}_{}.h5'.format(file, label_file)
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

    return None


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
            save_tsv(el[0], output, el[1], 'a')
    for el in data:
        preprocess_combined(el[0], tmps, el[1], file)


def load(filename):
    f = h5py.File(filename, 'r')
    data = {}

    for key in f:
        data[key] = f[key][...]
    f.close()

    return data


def save(filename, data):
    f = h5py.File(filename, 'w')

    for key in data:
        f.create_dataset(key, data[key].shape, dtype=data[key].dtype,
                         compression='gzip')[...] = data[key]

    f.close()


def get_set(folder, output, label):
    filelist = [os.path.join(folder, el) for el in os.listdir(folder)]
    data = OrderedDict()

    for f in filelist:
        data[f] = load(f)

    out_file = os.path.join(output, '{}_combined.h5'.format(label))
    save(out_file, merge_data(data))


def do_combined_preprocess(features, output, tsv_flag, mem_efficient, cpus):

    if mem_efficient:
        tmp_folder = os.path.join(os.path.dirname(features), 'tmp/')
        tmp_train = os.path.join(os.path.dirname(features), 'train/')
        tmp_test = os.path.join(os.path.dirname(features), 'test/')
        tmp_val = os.path.join(os.path.dirname(features), 'val/')

        print('Splitting original file...')
        os.mkdir(tmp_folder); os.mkdir(tmp_train); os.mkdir(tmp_test); os.mkdir(tmp_val)
        cmd = 'split -l {} {} {}'.format(20000, features, tmp_folder)
        subprocess.call(cmd, shell=True)

        print('Extracting features to h5 and tsv files...')
        counter = 0
        f = functools.partial(split_sets_files, tmp_folder=tmp_folder, \
                counter=counter, tsv_flag=tsv_flag, output=output, \
                    tmps=os.path.dirname(features))
        with Pool(cpus) as p:
            for i, rval in enumerate(p.imap_unordered(f, os.listdir(tmp_folder))):
                counter += 1

        print('Concatenating features into h5s...')
        get_set(tmp_test, output, 'test')
        get_set(tmp_val, output, 'val')
        get_set(tmp_train, output, 'train')

        print('Removing tmp folders and done')
        subprocess.call('rm -r {}'.format(tmp_folder), shell=True)
        subprocess.call('rm -r {}'.format(tmp_train), shell=True)
        subprocess.call('rm -r {}'.format(tmp_test), shell=True)
        subprocess.call('rm -r {}'.format(tmp_val), shell=True)

    else:
        df = pd.read_csv(features, sep='\t', names=names_all)
        data = get_training_test_val(pd.concat([treat, untreat]))

        if tsv_flag:
            for el in data:
                save_tsv(el[0], output, el[1])
        for el in data:
            preprocess_sequence(el[0], output, el[1])


def preprocess_err_read(err_treated, err_untreated, output):

    treat, untreat = get_data(err_treated, err_untreated,
        names=['read_name', 'pos', 'chr', 'k_mer', 'qual', 'mis', 'ins', 'del', 'methyl_label'])
    data = get_training_test_val(pd.concat([treat, untreat]))

    for el in data:
        df = el[0]
        file = el[1]
        kmer = df['k_mer'].apply(kmer2code)
        base_qual = [tf.strings.to_number(i.split(','), tf.float32) \
            for i in df['qual'].values]
        base_mis = [tf.strings.to_number(i.split(','), tf.float32) \
            for i in df['mis'].values]
        base_ins = [tf.strings.to_number(i.split(','), tf.float32) \
            for i in df['ins'].values]
        base_del = [tf.strings.to_number(i.split(','), tf.float32) \
            for i in df['del'].values]
        label = df['methyl_label']

        file_name = os.path.join(output, '{}_err_read.h5'.format(file))

        with h5py.File(file_name, 'a') as hf:
            hf.create_dataset("kmer",  data=np.stack(kmer))
            hf.create_dataset('qual',  data=np.stack(base_qual))
            hf.create_dataset('mis',  data=np.stack(base_mis))
            hf.create_dataset('ins',  data=np.stack(base_ins))
            hf.create_dataset('del',  data=np.stack(base_del))
            hf.create_dataset('methyl_label',  data=label)

    return None
