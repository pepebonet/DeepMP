#!/usr/bin/env python3

import click
import numpy as np
import pandas as pd
from collections import Counter

from bgreference import refseq

chr_dict = {'chrI': 1, 'chrII': 2, 'chrIII': 3, 'chrIV': 4, 'chrV': 5, 
    'chrVI': 6, 'chrVII': 7, 'chrVIII': 8, 'chrIX': 9, 'chrX': 10, 'chrXI': 11, 
    'chrXII': 12, 'chrXIII': 13, 'chrXIV': 14, 'chrXV': 15, 'chrXVI': 16, 
    'chrM': 17, 'chrmt' : 17}


def num2chr(df):
    df['Chromosome'] = df['Peak chr'].apply(lambda x : find_chr(x))
    return df


def find_chr(x):
    for k, v in chr_dict.items():
        if v == int(x.split('chr')[1]): 
            return k


def obtain_context(df, cont, cent):
    try: 
        seq = refseq('saccer3', df['Chromosome'], 
            df['Peak genomic coordinate'] - cent, cont)
    except: 
        seq = '-'
    
    return seq


def filter_sites(df):
    df['kmer'] = df['PENTAMER'].apply(lambda x: list(x))
    bases = pd.DataFrame(
        df['kmer'].values.tolist(),columns=['b1', 'b2', 'b3', 'b4', 'b5']
    )
    data = df.join(bases)
    data = apply_filters()
    import pdb; pdb.set_trace()


def extract_sites(df):
    df = num2chr(df)
    df['PENTAMER'] = df.apply(obtain_context, args=(5,2), axis=1)
    df = df[df['PENTAMER'] != '-']

    sites_filtered = filter_sites(df)
    #TODO <JB> Find sites that are in RRACH motif and check the total number
    #can be more but not less than 363 
    


@click.command(short_help='Obtain in vivo methylated sites from epinano')
@click.option(
    '-i', '--input', default='', help='File containing list of genes'
    'default=mmc1.csv'
)
@click.option(
    '-o', '--output', default='', help='output path for the generated sites'
)
def main(input, output):
    genes = pd.read_csv(input, sep='\t')
    import pdb;pdb.set_trace()
    extract_sites(genes)


if __name__ == "__main__":
    main()

