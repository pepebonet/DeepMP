#!/usr/bin/envs python3 

import os
import click
import pandas as pd
from sklearn.model_selection import train_test_split

import deepmp.utils as ut


names_all=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname',
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_skew', 'signal_kurt', 'signal_diff', 'signal_lens',
            'cent_signals', 'qual', 'mis', 'ins', 'del', 'methyl_label', 'flag']


def get_training_test_val(df):
    train, val = train_test_split(df, test_size=0.05, random_state=0)
    return [(train, 'train'), (val, 'val')]


def save_tsv(df, output, file, mode='w'):
    file_name = os.path.join(output, '{}.tsv'.format(file))
    if mode == 'a':
        df.to_csv(file_name, sep='\t', index=None, mode=mode, header=None)
    else:
        df.to_csv(file_name, sep='\t', index=None, mode=mode)
        

@click.command(short_help='')
@click.argument('input')
@click.option(
    '-o', '--output', default='', help='output folder'
)
def main(input, output):
    
    df = pd.read_csv(input, sep='\t', names=names_all)
    file = input.rsplit('.tsv')[0]
    import pdb;pdb.set_trace()

    data = get_training_test_val(df)

    for el in data:
        save_tsv(el[0], output, el[1], 'a')

    for el in data:
        if el[0].shape[0] > 0:
            ut.preprocess_combined(el[0], '', el[1], file)
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()