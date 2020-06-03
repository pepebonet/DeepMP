#!/usr/bin/env python3
import os
import click
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
    test = merged[merged['class'] == 1].drop(['class'], axis=1)
    remaining = merged[merged['class'] != 1].drop(['class'], axis=1)

    return test, test_positions, remaining


def get_positions(df, positions):
    df['id'] = df[0] + '_' + df[1].astype(str)
    merged = pd.merge(df, positions, on='id', how='outer')

    test = merged[merged['class'] == 1].drop(['class'], axis=1)
    remaining = merged[merged['class'] != 1].drop(['class'], axis=1)

    return remaining, test[test[11].notna()]


def save_int_outputs(df, out_path):
    df.to_csv(out_path, sep='\t', index=False, header=None)


def save_outputs(train, val, test, output, label):
    train.to_csv(
        os.path.join(output, "train_{}.tsv"), sep='\t', index=False, header=None
    )
    val.to_csv(
        os.path.join(output, "val.tsv"), sep='\t', index=False, header=None
    )
    test.to_csv(
        os.path.join(output, "test.tsv"), sep='\t', index=False, header=None
    )


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
    test_reads, val_reads, train_chunks = get_data(
        train_path, val_path, test_path
    )
    counter = 0
    test, positions, train = get_test_positions(test_reads, number_positions)

    val, test_from_val = get_positions(val_reads, positions)
    test = pd.concat([test, test_from_val])

    tmp_test, tmp_train = create_tmp_folders(output)
    tmp = os.path.join(output, 'tmp')
    if not os.path.isdir(tmp): 
        os.mkdir(tmp)

    tmp_test = 
    import pdb;pdb.set_trace()
    for chunk in train_chunks:
        #TODO we could save each train file at a time
        train_chunk, test_from_train = get_positions(chunk, positions)
        train = pd.concat([train, train_chunk])
        test = pd.concat([test, test_from_train])

    save_outputs(train.drop(['id'], axis=1), val.drop(['id'], axis=1), 
        test.drop(['id'], axis=1), output)



if __name__ == "__main__":
    main()
