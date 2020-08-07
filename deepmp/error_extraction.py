#!/usr/bin/env python3
import os
import click
import numpy as np
import pandas as pd
from collections import Counter


def get_data(error_features, label):
    errors = pd.read_csv(error_features, sep=',')
    errors['label'] = label
    return errors


def get_motif_data(df, kmer):
    if len(kmer) == 1:
        return df[df['base3'] == kmer]
    elif len(kmer) == 2:
        return df[(df['base3'] == kmer[0]) & (df['base4'] == kmer[1])]
    elif len(kmer) == 3:
        return df[(df['base2'] == kmer[0]) & (df['base3'] == kmer[1]) \
            & (df['base4'] == kmer[2])]
    elif len(kmer) == 4:
        return df[(df['base2'] == kmer[0]) & (df['base3'] == kmer[1]) \
            & (df['base4'] == kmer[2]) & (df['base5'] == kmer[3])]
    elif len(kmer) == 5:
        return df[df['#Kmer'] == kmer]
    else:
        raise NotImplementedError(
            'Epinano feature extraction accepts motifs of up to pentamer length'
        )


def clean_df(df):
    return df.drop(columns=['#Kmer', 'Window', 'Coverage',
        'base1', 'base2', 'base3', 'base4', 'base5', 'kmer',
        'pos1', 'pos2', 'pos4', 'pos5']).rename(columns={'pos3': 'pos'})
    

def get_modified_sites(df, kmer):
    df['kmer'] = df['#Kmer'].apply(lambda x: list(x))

    bases = pd.DataFrame(df['kmer'].values.tolist(), 
        columns=['base1', 'base2', 'base3', 'base4', 'base5'])

    try: 
        positions = pd.DataFrame(
            df['Window'].str.split(':').values.tolist(), 
            columns=['pos1', 'pos2', 'pos3', 'pos4', 'pos5']
        )
    except: 
        positions = pd.DataFrame(
            df['window'].str.split(':').values.tolist(), 
            columns=['pos1', 'pos2', 'pos3', 'pos4', 'pos5']
        )

    data = df.join(bases.join(positions))

    return clean_df(get_motif_data(data, kmer))


def build_kmer_features(df, meth_label):
    df['groups'] = np.arange(len(df)) // 5
    
    all_features = []; counter = 0
    for i, j in df.groupby('groups'):
        features = list(np.concatenate(
            [j['q_mean'].values, j['mis'].values, j['del'].values]
        ))
        features.append(meth_label)
        features.append(j['pos'].unique()[0])
        features.append(j['Ref'].unique()[0])

        all_features.append(features)

    return pd.DataFrame(all_features, columns=['q1', 'q2', 'q3', 'q4', 'q5', 
        'mis1', 'mis2', 'mis3', 'mis4', 'mis5', 'del1', 'del2', 
        'del3', 'del4', 'del5', 'label', 'pos', 'chr'])


def save_files(df, output, label):
    if label == '1': 
        df.to_csv(os.path.join(output, 'error_features_treat.csv'), index=None)
    else:
        df.to_csv(os.path.join(output, 'error_features_untreat.csv'), index=None)


def process_error_features(error_features, label, motif, output, memory_efficient):
    
    if memory_efficient:
        errors_kmer = pd.DataFrame()
        df = pd.read_csv(error_features, sep=',', nrows=1000000)
        names = df.columns

        counter = 0
        while df.shape[0] > 0:
            
            if 'Relative_pos' in df.columns:
                df = df.rename(columns={'window':'Window', 'cov':'Coverage'})
                df_kmer = get_modified_sites(df, motif)
                errors_int = build_kmer_features(df_kmer, label)
            else: 
                errors_int = get_modified_sites(df, motif)
            errors_kmer = pd.concat([errors_kmer, errors_int])

            counter += 1; print(counter)

            df = pd.read_csv(error_features, sep=',', nrows=1000000, 
                skiprows=(1000000 * counter) + 1, names=names)

    else: 
        errors = get_data(error_features, label)
        error_kmer = get_modified_sites(errors, motif)

    save_files(error_kmer, output, label)
