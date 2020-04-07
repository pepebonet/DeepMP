#!/usr/bin/env python3
import os
import click
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split


def get_data(error_features, sequence_features):
    errors = pd.read_csv(error_features, sep=',')
    sequence = pd.read_csv(sequence_features, delimiter = "\t", 
        names = ['chrom', 'pos', 'strand', 'pos_in_strand', 'readname',
        'read_strand', 'kmer','signal_means', 'signal_stds', 'signal_lens', 
        'cent_signals', 'methy_label'])

    errors['label'] = 1

    return errors, sequence


#TODO <JB> automatize for every kmer possible and clean
def get_modified_sites(df, kmer):
    df['kmer'] = df['#Kmer'].apply(lambda x: list(x))
    bases = pd.DataFrame(df['kmer'].values.tolist(), 
        columns=['base1', 'base2', 'base3', 'base4', 'base5'])
    positions = pd.DataFrame(
        df['Window'].str.split(':').values.tolist(), 
        columns=['pos1', 'pos2', 'pos3', 'pos4', 'pos5']
    )
    df2 = df.join(bases.join(positions))

    return df2[(df2['base3'] == 'C') & (df2['base4'] == 'G')]


def get_merge_data(errors, sequence):
    import pdb;pdb.set_trace()


def get_training_test_data(df):
    X = df[df.columns[:-1]]
    Y = df[df.columns[-1]]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.05, random_state=0
    )
    X_test['label'] = Y_test

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.01, random_state=0
    )
    X_train['label'] = Y_train; X_val['label'] = Y_val

    return X_train, X_test, X_val


def save_files(train, test, val, output):
    train.to_csv(os.path.join(output, 'train_errors.csv'), index=None)
    test.to_csv(os.path.join(output, 'test_errors.csv'), index=None)
    val.to_csv(os.path.join(output, 'val_errors.csv'), index=None)


# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='Merge error and sequence features')
@click.option(
    '-ef', '--error-features', default='', help='extracted error features'
)
@click.option(
    '-sf', '--sequence-features', default='', help='extracted sequence features'
)
@click.option(
    '-l', '--label', default='treat', type=click.Choice(['treat', 'untreat']),
)
@click.option(
    '-m', '--motif', default='GC', help='motif of interest'
)
@click.option(
    '-o', '--output', default='', help='Output file'
)
def main(error_features, sequence_features, label, motif, output):
    errors, sequence = get_data(error_features, sequence_features)
    
    error_kmer = get_modified_sites(errors, motif)

    sequence_error = get_merge_data(error_kmer, sequence)

    train, test, val = get_training_test_data(sequence_error)

    save_files(train, test, val, output)
    import pdb;pdb.set_trace()


    
if __name__ == "__main__":
    main()