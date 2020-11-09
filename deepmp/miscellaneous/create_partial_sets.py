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
    sets = [(5, 95), (10, 90), (25, 75), (50,50), (75, 25), (90, 10), (95, 10)]

    feats = pd.read_csv(features, header=None, sep='\t', names=names_all)

    treated = feats[feats['methyl_label'] == 1]
    untreated = feats[feats['methyl_label'] == 0]

    treat_shape = treated.shape[0]
    for el in tqdm(sets): 
        sample_treat = treated.sample(int(round(treat_shape * el[0] / 100, 0)))
        sample_untreat = untreated.sample(int(round(treat_shape * el[1] / 100, 0)))
        partial_set = pd.concat([sample_treat, sample_untreat]).sample(frac=1).reset_index(drop=True)
        out_file = os.path.join(output, 'treat_{}_untreat_{}.tsv'.format(str(el[0]), str(el[1])))
        partial_set.to_csv(out_file, sep='\t', header=None)
        ut.preprocess_combined(partial_set, output, 'untreat_{}'.format(el[1]), 'treat_{}'.format(el[0]))

    
    import pdb;pdb.set_trace()


if __name__ == "__main__":
    main()