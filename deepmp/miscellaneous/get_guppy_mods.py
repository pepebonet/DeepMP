#!/usr/bin/envs python3

#!/usr/bin/envs python3

import os 
import sys
import click
import numpy as np 
import pandas as pd 
from tqdm import tqdm 

sys.path.append('../')
import deepmp.utils as ut


def split_multiple_cpgs(record):
    cpgs = []; static = record.values[3:]
    chrom = str(record['chromosome'])
    start = int(record['start'])
    end = int(record['end'])
    # find the position of the first CG dinucleotide
    sequence = record['sequence']
    cg_pos = sequence.find("CG")
    first_cg_pos = cg_pos
    
    while cg_pos != -1:
        key = [chrom, start + cg_pos - first_cg_pos, start + cg_pos - first_cg_pos]
        cpgs.append(np.concatenate([key, static]))
        cg_pos = sequence.find("CG", cg_pos + 1)


    return list(cpgs)


def get_filters_and_probs(guppy):

    guppy['prob_meth'] = np.exp(guppy['log_lik_ratio']) \
         / (1 + np.exp(guppy['log_lik_ratio']))
    guppy['prob_unmeth'] = 1 \
         / (1 + np.exp(guppy['log_lik_ratio']))

    return guppy

    
def get_labels(df):

    infer_pos = df[df['prob_meth'] >= 0.5]
    infer_neg = df[df['prob_meth'] < 0.5]
    infer_pos['Prediction'] = 1
    infer_neg['Prediction'] = 0

    return pd.concat([infer_neg, infer_pos])


def get_readname_column(df, dict):
    df['readname'] = dict[df['read_name'] + '.txt']
    return df


def get_positions_only(df, positions):
    # import pdb;pdb.set_trace()
    df = pd.merge(
        df, positions, right_on=['chr', 'start', 'strand'], 
        left_on=['#chromosome', 'start', 'strand']
    )
    try:
        label = np.zeros(len(df), dtype=int)
        label[np.argwhere(df['status'].values == 'mod')] = 1

        df['methyl_label'] = label

    except: 
        pass

    return df


@click.command(short_help='get modifications from guppy')
@click.option(
    '-go', '--guppy_output', required=True, 
    help='call methylation output from guppy'
)
@click.option(
    '-dr', '--dict_reads', default='', help='dictionary with readnames'
)
@click.option(
    '-p', '--positions', default='', help='position to filter out'
)
@click.option(
    '-o', '--output', required=True, 
    help='Path to save modifications called'
)
def main(guppy_output, dict_reads, positions, output):

    guppy = pd.read_csv(guppy_output, sep='\t')
    
    guppy = get_filters_and_probs(guppy)

    df_calling = get_labels(guppy)

    if positions:
        positions = pd.read_csv(positions, sep='\t')
        df_calling = get_positions_only(df_calling, positions)
    
    if dict_reads: 
        dict_names = ut.load_obj(dict_reads)  
        dict_names = {v: k for k, v in dict_names.items()}
        aa = df_calling['read_name'] + '.txt'
        bb = [dict_names[x] for x in aa.tolist()]
        df_calling['readnames'] = bb

    df_calling.to_csv(
        os.path.join(output, 'mods_guppy_readnames.tsv'), sep='\t', index=None
    )


if __name__ == '__main__':
    main()