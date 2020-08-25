#!/usr/bin/env python3
import os
import click
import numpy as np
import pandas as pd
from collections import Counter


def error_features_kmer(df, meth_label):
    df['cent_base'] = df['#Kmer'].apply(lambda x: x[2])
    df['label'] = np.int(meth_label)
    return df[df['cent_base'] == 'A'].drop(columns=['cent_base'])
    

def error_features_single(df, meth_label):
    data = df[(df['relative_pos'] == 0) & (df['Ref_base'] == 'A')]
    data = data[['mean_q', 'mis', 'ins', 'del']]
    data['label'] = meth_label
    return data


def save_output_errors(df, input, output, methyl_label):
    if methyl_label == '1':
        status = 'MOD'
    else:
        status = 'UNM'
    df['label'] = df['label'].astype('int32')
    out_file = os.path.join(output, 'error_features_{}.csv'.format(status))
    df.to_csv(out_file, index=None)
    

# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='Arrange features epinano data')
@click.option(
    '-in', '--inputs', multiple=True, required=True
)
@click.option(
    '--methyl-label', '-ml', type=click.Choice(['1', '0']), 
    default='1', help='the label of the interested modified '
    'bases, this is for training. 0 or 1, default 1'
)
@click.option(
    '--feature_option', '-fo', type=click.Choice(['kmer', 'single']), 
    default='kmer'
)
@click.option(
    '-o', '--output', default='', 
    help='Output file extension'
)
def main(inputs, methyl_label, feature_option, output):
    data = pd.DataFrame()

    for path in inputs:
        input_n = pd.read_csv(path, sep=',')
        if 'q0' in input_n.columns:
            input_n = input_n.rename(columns={"q0": "q1"})
        replicate = path.rsplit('/', 1)[-1].split('.')[0]
        data = pd.concat([data, input_n], sort=False)
    
    if feature_option == 'single':
        errors = error_features_single(data.copy(), int(methyl_label))
    else:
        errors = error_features_kmer(data.copy(), int(methyl_label))

    save_output_errors(errors, input, output, methyl_label)


if __name__ == "__main__":
    main()