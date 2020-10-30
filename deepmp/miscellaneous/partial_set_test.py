#!usr/bin/envs python3 

import os
import sys
import click 
import functools
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from random import random
from collections import Counter
from multiprocessing import Pool
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support

sys.path.append('../')
import deepmp.utils as ut
import deepmp.preprocess as pr

names_all=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname',
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_skew', 'signal_kurt', 'signal_diff', 'signal_lens',
            'cent_signals', 'qual', 'mis', 'ins', 'del', 'methyl_label', 'flag']


def acc_test_single(data, labels, model_file, score_av='binary'):
    model = load_model(model_file)
    
    pred =  model.predict(data).flatten()
    inferred = np.zeros(len(pred), dtype=int)
    inferred[np.argwhere(pred >= 0.5)] = 1

    precision, recall, f_score, _ = precision_recall_fscore_support(
        labels, inferred, average=score_av
    )

    return [precision, recall, f_score], pred, inferred


def process_chunk(features, tmp_folder, output, model):
    df = pd.read_csv(os.path.join(tmp_folder, features), sep='\t', names=names_all)
    df = df[(df['pos'] >= 1000000) & (df['pos'] <= 2000000)]
    counter = round(random(), 10)

    if df.shape[0] > 0:
        ut.preprocess_combined(df, output, 'all_{}'.format(counter), 'test')
        test_file = os.path.join(output, 'test_all_{}.h5'.format(counter))
        
        data_seq, data_err, labels = ut.get_data_jm(test_file, 17)
        
        acc, pred, inferred = acc_test_single([data_seq, data_err], labels, model)

        info  = Counter(df['methyl_label'])
        print(acc, info)


@click.command(short_help='script to test bug in partial sets')
@click.option(
    '-f', '--features', help='features path', required=True
)
@click.option(
    '-m', '--model', required=True, help='trained model'
)
@click.option(
    '-cpu', '--cpus', default=1, help='number of cpus'
)
@click.option(
    '-o', '--output', help='output path'
)
def main(features, model, cpus, output):

    tmp_folder = os.path.join(os.path.dirname(features), 'tmp_test_partials/')

    print('Splitting original file...')
    os.mkdir(tmp_folder)
    cmd = 'split -l {} {} {}'.format(100000, features, tmp_folder)
    subprocess.call(cmd, shell=True)
    
    print('Executing...')
    f = functools.partial(process_chunk, tmp_folder=tmp_folder, output=output, \
                model=model)
    with Pool(cpus) as p:
        for i, rval in enumerate(p.imap_unordered(f, os.listdir(tmp_folder))):
            pass
    

if __name__ == "__main__":
    main()