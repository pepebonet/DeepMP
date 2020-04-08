#!/usr/bin/env python3
import os
import click
import pandas as pd


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
    return df.drop(columns=['#Kmer', 'Window', 'Ref', 'Coverage',
        'base1', 'base2', 'base3', 'base4', 'base5', 'kmer',
        'pos1', 'pos2', 'pos4', 'pos5']).rename(columns={'pos3': 'pos'})
    

def get_modified_sites(df, kmer):
    df['kmer'] = df['#Kmer'].apply(lambda x: list(x))

    bases = pd.DataFrame(df['kmer'].values.tolist(), 
        columns=['base1', 'base2', 'base3', 'base4', 'base5'])

    positions = pd.DataFrame(
        df['Window'].str.split(':').values.tolist(), 
        columns=['pos1', 'pos2', 'pos3', 'pos4', 'pos5']
    )

    data = df.join(bases.join(positions))

    return clean_df(get_motif_data(data, kmer))


def save_files(df, output, label):
    if label == '1': 
        df.to_csv(os.path.join(output, 'error_features_treat.csv'), index=None)
    else:
        df.to_csv(os.path.join(output, 'error_features_untreat.csv'), index=None)


def process_error_features(error_features, label, motif, output):
    errors = get_data(error_features, label)
    
    error_kmer = get_modified_sites(errors, motif)

    save_files(error_kmer, output, label)
