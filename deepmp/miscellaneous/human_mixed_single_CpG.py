#!/usr/bin/env python3

import os 
import sys
import click
import itertools
import numpy as np
import pandas as pd
from itertools import islice
from collections import Counter

sys.path.append('../')
import deepmp.utils as ut

names_all=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname',
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_skew', 'signal_kurt', 'signal_diff', 'signal_lens',
            'cent_signals', 'qual', 'mis', 'ins', 'del', 'methyl_label']


def slicing_window(seq, n):

    it = iter(seq)
    result = ''.join(islice(it, n))

    if len(result) == n:
        yield result

    for elem in it:
        result = result[1:] + elem
        yield result

    
def get_count_CpG(seq):
    return Counter(list(slicing_window(seq, 2)))['CG']


# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='Separate mixid and single CpG human')
@click.option(
    '-tf', '--test_file', required=True,
    help='test.tsv to separate'
)
@click.option(
    '-o', '--output', default='', 
    help='Output file extension'
)
def main(test_file, output):
    # import pdb;pdb.set_trace()
    df = pd.read_csv(test_file, sep='\t', header=None, names=names_all)
    df['CpG_Counter'] = df['kmer'].apply(get_count_CpG)

    single = df[df['CpG_Counter'] == 1]
    mixed = df[df['CpG_Counter'] > 1]

    single.to_csv(os.path.join(output, 'single_test.tsv'), sep='\t', index=None)
    mixed.to_csv(os.path.join(output, 'mixed_test.tsv'), sep='\t', index=None)

    ut.preprocess_combined(single, output, 'test', 'single')
    ut.preprocess_combined(mixed, output, 'test', 'mixed')


if __name__ == '__main__':
    main()
