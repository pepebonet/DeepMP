#!/usr/bin/env python3
import os
import click
import random
import subprocess
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def get_data(train_path, val_path, test_path):
    train_chunks = pd.read_csv(train_path, sep='\t', header=None, chunksize=1000000)
    test = pd.read_csv(test_path, sep='\t', header=None)
    val = pd.read_csv(val_path, sep='\t', header=None)
    
    return test, val, train_chunks


def get_test_positions(df, num_pos):
    df['id'] = df[0] + '_' + df[1].astype(str)
    treated = df[df[11] == 1]; untreated = df[df[11] == 0]
    treat_pos = treated['id'].unique()[:int(num_pos/2)]
    untreat_pos = untreated['id'].unique()[:int(num_pos/2)]

    all_pos = np.concatenate([treat_pos, untreat_pos])
    distinct = np.ones(len(all_pos))
    test_positions = pd.DataFrame(
        {'id': all_pos, 'class': distinct}, columns=['id', 'class']
    )

    merged = pd.merge(df, test_positions, on='id', how='outer')
    test = merged[merged['class'] == 1].drop(['class', 'id'], axis=1)
    remaining = merged[merged['class'] != 1].drop(['class', 'id'], axis=1)

    return test, test_positions, remaining


def get_positions(df, positions):
    df['id'] = df[0] + '_' + df[1].astype(str)
    merged = pd.merge(df, positions, on='id', how='outer')
    
    test = merged[merged['class'] == 1].drop(['class', 'id'], axis=1).dropna()
    test[11] = test[11].astype(int)

    remaining = merged[merged['class'] != 1].drop(['class', 'id'], axis=1).dropna()
    remaining[11] = remaining[11].astype(int)

    assert test.shape[0] == pd.merge(df, positions, on='id', how='inner').shape[0]
    return remaining, test


def save_int_outputs(df, out_path):
    df.to_csv(out_path, sep='\t', index=False, header=None)


def create_tmp_folders(output):
    tmp = os.path.join(output, 'tmp')
    if not os.path.isdir(tmp): 
        os.mkdir(tmp)

    tmp_test = os.path.join(tmp, 'test')
    if not os.path.isdir(tmp_test): 
        os.mkdir(tmp_test)

    tmp_train = os.path.join(tmp, 'train')
    if not os.path.isdir(tmp_train): 
        os.mkdir(tmp_train)

    return tmp_test, tmp_train


def concat_files(in_path, out_path):
    cmd = 'cat {} > {}'. format(in_path, out_path)
    subprocess.call(cmd, shell=True)
    subprocess.call('rm -r {}'.format(os.path.dirname(in_path)), shell=True)


def split_sets(train_path, val_path, test_path, output, number_positions):
    tmp_test, tmp_train = create_tmp_folders(output)
        
    test_reads, val_reads, train_chunks = get_data(
        train_path, val_path, test_path
    )

    test, positions, train = get_test_positions(test_reads, number_positions)

    save_int_outputs(train, os.path.join(tmp_train, 'train_0.tsv'))

    val, test_from_val = get_positions(val_reads, positions)
    save_int_outputs(val, os.path.join(output, 'val.tsv'))

    test = pd.concat([test, test_from_val])
    save_int_outputs(test, os.path.join(tmp_test, 'test_0.tsv'))

    counter = 1
    for chunk in train_chunks:
        train_chunk, test_from_train = get_positions(chunk, positions)

        save_int_outputs(
            train_chunk, os.path.join(tmp_train, 'train_{}.tsv'.format(counter))
        )
        save_int_outputs(
            test_from_train, os.path.join(tmp_test, 'test_{}.tsv'.format(counter))
        )
        counter += 1

    concat_files(os.path.join(tmp_test, '*.tsv'), os.path.join(output, 'test.tsv'))
    concat_files(os.path.join(tmp_train, '*.tsv'), os.path.join(output, 'train.tsv'))


def split_treatments(method, treated_path, untreated_path, output, positions_list):
    tmp_test, tmp_train = create_tmp_folders(output)
    treat_chunks, untreat_chunks = get_data_treatments(
        treated_path, untreated_path
    )

    if positions_list:
        positions = pd.read_csv(positions_list, sep='\t')
        positions['class'] = 1
    else:
        raise NotImplementedError('A lost of positions needs to be given')

    iterate_chunks(
        treat_chunks, positions, tmp_train, tmp_test, output, label='treat'
    )
    iterate_chunks(
        untreat_chunks, positions, tmp_train, tmp_test, output, label='untreat'
    )

    concat_val(output)
    concat_files(os.path.join(tmp_test, '*.tsv'), os.path.join(output, 'test.tsv'))
    concat_files(os.path.join(tmp_train, '*.tsv'), os.path.join(output, 'train.tsv'))
    subprocess.call('rm -r {}'.format(os.path.join(output, 'tmp')), shell=True)


def get_data_treatments(treat, untreat):
    treated = pd.read_csv(treat, sep='\t', header=None, chunksize=1000000)
    untreated = pd.read_csv(untreat, sep='\t', header=None, chunksize=1000000)
    return treated, untreated


def iterate_chunks(chunks, positions, tmp_train, tmp_test, output, label='all'):
    counter = 1
    for chunk in chunks:
        train_chunk, test_from_train = get_positions(chunk, positions)
        train_chunk = shuffle(train_chunk)
        test_from_train = shuffle(test_from_train)

        if counter == 1 and label != 'all':
            val = train_chunk.sample(n=5000)
            save_int_outputs(val, os.path.join(output, 'val_{}.tsv'.format(label)))

        save_int_outputs(
            train_chunk, os.path.join(tmp_train, 'train_{}_{}.tsv'.format(counter, label))
        )
        save_int_outputs(
            test_from_train, os.path.join(tmp_test, 'test_{}_{}.tsv'.format(counter, label))
        )
        counter += 1


def concat_val(output):
    out_path = os.path.join(output, 'val.tsv')
    treat_path = os.path.join(output, 'val_treat.tsv')
    untreat_path = os.path.join(output, 'val_untreat.tsv')
    cmd = 'cat {} {} > {}'. format(treat_path, untreat_path, out_path)
    subprocess.call(cmd, shell=True)
    subprocess.call('rm {}'.format(treat_path), shell=True)
    subprocess.call('rm {}'.format(untreat_path), shell=True)


@click.command(short_help='Script to separate files per position')
@click.option(
    '-m', '--method', required=True,
    type=click.Choice(['sets', 'treatments']),
    help='Choose type of input files. Either train, test and val sets (sets) '
    'or treated and untreated (treatments'
)
@click.option(
    '-tr', '--train-path', help='Train file'
)
@click.option(
    '-va', '--val-path', help='Validation file'
)
@click.option(
    '-te', '--test-path', help='Test file'
)
@click.option(
    '-tp', '--treated-path', help='treated file'
)
@click.option(
    '-unp', '--untreated-path', help='untreated file'
)
@click.option(
    '-np', '--number-positions', default=30000, 
    help='number of positions to be selected for the test set'
)
@click.option(
    '-pl', '--positions-list', default='',
    help='List of position to store for the test set'
)
@click.option(
    '-o', '--output', default=''
)
def main(method, train_path, val_path, test_path, treated_path, untreated_path,
    output, number_positions, positions_list):

    if method == 'sets':
        split_sets(train_path, val_path, test_path, output, number_positions)
    else:
        split_treatments(method, treated_path, untreated_path, output, positions_list)


if __name__ == "__main__":
    main()
