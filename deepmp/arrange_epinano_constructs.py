#!/usr/bin/env python3
import os
import click
import numpy as np
import pandas as pd
from collections import Counter


def error_features_kmer(df, meth_label):
    df['groups'] = np.arange(len(df)) // 5
    
    all_features = []; counter = 0
    for i, j in df.groupby('groups'):
        if j[j['relative_pos'] == 0]['Ref_base'].tolist()[0] == 'A':
            if Counter(j['Ref_base'])['A'] > 1:
                counter += 1
            features = np.concatenate(
                [j['mean_q'].values, j['mis'].values, j['del'].values]
            )
            features = np.append(features, meth_label)

            all_features.append(features)

    return pd.DataFrame(all_features, columns=['q1', 'q2', 'q3', 'q4', 'q5', 
        'mis1', 'mis2', 'mis3', 'mis4', 'mis5', 'del1', 'del2', 
        'del3', 'del4', 'del5', 'label'])


def error_features_single(df, meth_label):
    data = df[(df['relative_pos'] == 0) & (df['Ref_base'] == 'A')]
    data = data[['mean_q', 'mis', 'ins', 'del']]
    data['label'] = meth_label
    return data


def sequence_features(df, methyl_label):
    df = df[df['relative_pos'] == 0]
    features = []
    for i in range(len(df) - 17):
        sub = df[i:i+17]
        if sub['Ref_base'][8:9].tolist()[0] == 'A':
            kmer = ''.join([x for x in sub['Ref_base'].tolist()])
            means_text = ','.join(
                [str(x) for x in np.around(sub['mean_current'].tolist(), 
                decimals=6)]
            )
            stds_text = ','.join(
                [str(x) for x in np.around(sub['std_current'].tolist(), 
                decimals=6)]
            )
            signal_len_text = ','.join(
                [str(x) for x in sub['Depth'].astype('int32').tolist()]
            )
            features.append(
                "\t".join(['-', '-', '-', '-', '-', '-', 
                kmer, means_text, stds_text, signal_len_text, '-', methyl_label])
            )
    return features


def _write_featurestr_to_file(features_str, input, output):
    write_fp = os.path.join(output, 'sequence_features_{}'.format(
        input.rsplit('/', 1)[-1]
    ))
    with open(write_fp, 'w') as wf:
        for one_features_str in features_str:
            wf.write(one_features_str + "\n")
        wf.flush()


def save_output_errors(df, input, output):
    df['label'] = df['label'].astype('int32')
    out_file = os.path.join(output, 'error_features_{}'.format(
        input.rsplit('/', 1)[-1]
    ))
    df.to_csv(out_file, index=None)
    

# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='Arrange features epinano data')
@click.argument(
    'input'
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
def main(input, methyl_label, feature_option, output):
    data = pd.read_csv(input, sep=',')
    data = data[data['std_current'].notna()]

    if feature_option == 'single':
        errors = error_features_single(data.copy(), int(methyl_label))
    else:
        errors = error_features_kmer(data.copy(), int(methyl_label))
    save_output_errors(errors, input, output)

    sequence = sequence_features(data.copy(), methyl_label)
    _write_featurestr_to_file(sequence, input, output)

    
if __name__ == "__main__":
    main()