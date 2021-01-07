#!/usr/bin/envs python3
import os
import sys
import click
import pandas as pd

from tqdm import tqdm

sys.path.append('../')
import deepmp.utils as ut

names_all=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname',
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_median',
            'signal_skew', 'signal_kurt', 'signal_diff', 'signal_lens',
            'cent_signals', 'qual', 'mis', 'ins', 'del', 'methyl_label', 'flag']


@click.command(short_help='script to create partial sets')
@click.option(
    '-f', '--features', help='features path', required=True
)
@click.option(
    '-o', '--output', help='output path', default=''
)
def main(features, output):
    # sets = [(0, 100), (10, 90), (20, 80), (30, 70), (40, 60), (50,50), (60, 40), (70, 30), (80, 20), (90, 10), (100, 0)]
    # sets = [(0, 100), (1, 99), (2, 98), (3, 97), (4, 96), (5, 95), (6, 94), (7, 93), (8, 92), (9, 91), (10, 90)]
    sets = [(8, 92)]

    feats = pd.read_csv(features, header=None, sep='\t', names=names_all)

    treated = feats[feats['methyl_label'] == 1]
    untreated = feats[feats['methyl_label'] == 0]

    treat_shape = treated.shape[0]

    for el in tqdm(sets): 
        if el[0] == 0:
            partial_set = untreated.sample(int(round(treat_shape * el[1] / 100, 0)))
        elif el[1] == 0:
            partial_set = treated.sample(int(round(treat_shape * el[0] / 100, 0)))
        else:
            sample_treat = treated.sample(int(round(treat_shape * el[0] / 100, 0)))
            sample_untreat = untreated.sample(int(round(treat_shape * el[1] / 100, 0)))
            partial_set = pd.concat([sample_treat, sample_untreat]).sample(frac=1).reset_index(drop=True)

        out_file = os.path.join(output, 'treat_{}_untreat_{}.tsv'.format(str(el[0]), str(el[1])))
        partial_set.to_csv(out_file, sep='\t', header=None, index=None)
        ut.preprocess_combined(partial_set, output, 'untreat_{}'.format(el[1]), 'treat_{}'.format(el[0]))

    import pdb;pdb.set_trace()


if __name__ == "__main__":
    main()