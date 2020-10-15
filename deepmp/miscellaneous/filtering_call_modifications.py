#!/usr/bin/envs python3
import os
import sys
import click 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support

sys.path.append('../')
import deepmp.utils as ut
import deepmp.preprocess as pr

names_all=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_skew', 'signal_kurt', 'signal_diff', 'signal_lens', 
            'cent_signals', 'qual', 'mis', 'ins', 'del', 'methyl_label', 'flag']


def get_mean(df):
    results = list(map(int, df.split(',')))
    return np.round(np.mean(results), 2)


def preprocess_combined(df, kmer_len):
    
    kmer = df['kmer'].apply(ut.kmer2code)
    base_mean = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_means'].values]
    base_std = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_stds'].values]
    base_median = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_median'].values]
    base_skew = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_skew'].values]
    base_diff = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_diff'].values]
    base_signal_len = [tf.strings.to_number(i.split(','), tf.float32) \
        for i in df['signal_lens'].values]
    label = df['methyl_label']

    embedding_size = 5
    embedded_bases = tf.one_hot(np.stack(kmer), embedding_size)
    # import pdb;pdb.set_trace()
    ## prepare inputs for NNs
    data = ut.concat_tensors_seq(embedded_bases, np.stack(base_mean), \
        np.stack(base_std), np.stack(base_median), \
        np.stack(base_diff), np.stack(base_signal_len), kmer_len)
    
    return data, label


def acc_test_single(data, labels, model_file, score_av='binary'):
    model = load_model(model_file)
    test_loss, test_acc = model.evaluate(data, tf.convert_to_tensor(labels))

    pred =  model.predict(data).flatten()
    inferred = np.zeros(len(pred), dtype=int)
    inferred[np.argwhere(pred >= 0.5)] = 1

    precision, recall, f_score, _ = precision_recall_fscore_support(
        labels, inferred, average=score_av
    )
    
    return [test_acc, precision, recall, f_score], pred, inferred


@click.command(short_help='analyse filtered mods by number of signals')
@click.option('-h5f', '--h5_features', required=True, help='h5 file')
@click.option('-tf', '--tsv_features', required=True, help='tsv file')
@click.option('-cm', '--called_mods', required=True, help='tsv file')
@click.option('-km', '--kmer', default=17, help='kmer length')
@click.option('-mf', '--model_file', default='', help='model')
@click.option('-d', '--deepsignal', default='', help='deepsignal file')
@click.option('-o', '--output', default='', help='Path to save dict')
def main(h5_features, tsv_features, called_mods, kmer, model_file, deepsignal, output):
    data_seq, labels = ut.get_data_sequence(h5_features, kmer)
    test = pd.read_csv(tsv_features, sep='\t', names=names_all)
    called_mod = pd.read_csv(called_mods, sep='\t')

    # test['probs'] = called_mod['probs']
    # test['inferred'] = np.round(called_mod['probs'].values).astype(int)

    test['mean_num_signals'] = test['signal_lens'].apply(get_mean)

    # import pdb;pdb.set_trace()
    test_low = test[test['mean_num_signals'] <= 10]
    test_high = test[test['mean_num_signals'] > 10]
    print(test_low.shape)
    print(test_high.shape)

    if deepsignal:
        deepsignal = pd.read_csv(deepsignal, sep='\t', header=None)

        test_high['id'] = test_high['readname'] + '_' + test_high['pos'].astype(str)
        deepsignal['id'] = deepsignal[4] + '_' + deepsignal[1].astype(str)
        merge = pd.merge(deepsignal, test_high, on='id', how='inner') 
        precision, recall, f_score, _ = precision_recall_fscore_support(
            merge['methyl_label'].values, merge[8].values, average='binary'
        )
        print(precision, recall, f_score)
    
    data_low, labels_low = preprocess_combined(test_low, kmer)
    acc_low, pred_low, inferred_low = acc_test_single(data_low, labels_low, model_file)

    data_high, labels_high = preprocess_combined(test_high, kmer)
    acc_high, pred_high, inferred_high = acc_test_single(data_high, labels_high, model_file)

    print(acc_low, acc_high)


if __name__ == "__main__":
    main()