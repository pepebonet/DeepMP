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


def get_filters_and_probs(nanopolish):
    conf_pos = nanopolish[nanopolish['log_lik_ratio'] > 2.0]
    conf_neg = nanopolish[nanopolish['log_lik_ratio'] < -2.0]
    nanopolish = pd.concat([conf_pos, conf_neg])

    nanopolish['prob_meth'] = np.exp(nanopolish['log_lik_ratio']) \
         / (1 + np.exp(nanopolish['log_lik_ratio']))
    nanopolish['prob_unmeth'] = 1 \
         / (1 + np.exp(nanopolish['log_lik_ratio']))

    single_cpgs = nanopolish[nanopolish['num_cpgs'] == 1]
    multiple_cpgs = nanopolish[nanopolish['num_cpgs'] >= 2]

    return nanopolish, single_cpgs, multiple_cpgs

    
def get_labels(df):

    infer_pos = df[df['prob_meth'] >= 0.5]
    infer_neg = df[df['prob_meth'] < 0.5]
    infer_pos['Prediction'] = 1
    infer_neg['Prediction'] = 0

    return pd.concat([infer_neg, infer_pos])


def get_readname_column(df, dict):
    df['readname'] = dict[df['read_name'] + '.txt']
    return df


@click.command(short_help='get modifications from nanopolish')
@click.option(
    '-no', '--nanopolish_output', required=True, 
    help='call methylation output from Nanopolish'
)
@click.option(
    '-dr', '--dict_reads', default='', help='dictionary with readnames'
)
@click.option(
    '-o', '--output', required=True, 
    help='Path to save modifications called'
)
def main(nanopolish_output, dict_reads, output):

    nanopolish = pd.read_csv(nanopolish_output, sep='\t')

    nanopolish, single_cpgs, multiple_cpgs = get_filters_and_probs(nanopolish)
    
    output_cpgs = multiple_cpgs.apply(split_multiple_cpgs, axis=1)
    multiple_df = pd.DataFrame(
        np.vstack(output_cpgs.tolist()), columns=nanopolish.columns
    )

    df_calling = get_labels(pd.concat([single_cpgs, multiple_df]))

    if dict_reads: 
        dict_names = ut.load_obj(dict_reads)  
        dict_names = {v: k for k, v in dict_names.items()}
        aa = df_calling['read_name'] + '.txt'
        bb = [dict_names[x] for x in aa.tolist()]
        df_calling['readnames'] = bb
    import pdb;pdb.set_trace()
    df_calling.to_csv(
        os.path.join(output, 'mods_nanopolish_readnames.tsv'), sep='\t', index=None
    )


if __name__ == '__main__':
    main()