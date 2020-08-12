#!/usr/bin/env python3

import os 
import click
import numpy as np
import pandas as pd 

def get_motif_data(df):
    return df[((df['base1'] == 'A') | (df['base1'] == 'G')) & \
        ((df['base2'] == 'A') | (df['base2'] == 'G')) & \
        (df['base3'] == 'A') & (df['base4'] == 'C') & \
        ((df['base5'] == 'A') | (df['base5'] == 'C') | (df['base5'] == 'T'))]


def clean_df(df):
    return df.drop(columns=['Window',
        'base1', 'base2', 'base3', 'base4', 'base5', 'kmer',
        'pos1', 'pos2', 'pos4', 'pos5']).rename(columns={'pos3': 'pos'})


def get_coverage(df):
    try:
        return int(np.average([float(el) for el in df.split(':')]))
    except:
        return 0


def get_modified_sites(df):
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

    return clean_df(get_motif_data(data))


@click.command(short_help='Get the right features from epinano yeast data')
@click.option(
    '-f', '--features', required=True,
    help='Features of the modified samples'
)
@click.option(
    '-o', '--output', required=True,
    help='output path to save training and test files'
)
def main(features, output):
    df = pd.read_csv(features, sep=',')
    mod_sites = get_modified_sites(df)
    mod_sites['Cov'] = mod_sites['Coverage'].apply(get_coverage)
    mod_sites = mod_sites[mod_sites['Cov'] >=5].drop(columns=['Coverage'])

    mod_sites.to_csv(output, sep='\t', index=None)
    

if __name__ == "__main__":
    main()