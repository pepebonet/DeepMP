#!/usr/bin/env python3
import os
import click
import random
import subprocess
import numpy as np
import pandas as pd

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
    subprocess.call('rm -r {}'.format(in_path), shell=True)


@click.command(short_help='Script to separate files per position')
@click.option(
    '-tr', '--train-path', required=True, help='Train file'
)
@click.option(
    '-va', '--val-path', required=True, help='Validation file'
)
@click.option(
    '-te', '--test-path', required=True, help='Test file'
)
@click.option(
    '-np', '--number-positions', default=30000, 
    help='number of positions to be selected for the test set'
)
@click.option(
    '-o', '--output', default=''
)
def main(train_path, val_path, test_path, output, number_positions):
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

    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
