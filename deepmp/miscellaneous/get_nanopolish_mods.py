#!/usr/bin/envs python3

import os 
import sys
import click
import numpy as np 
import pandas as pd 
from tqdm import tqdm 

sys.path.append('../')
import deepmp.utils as ut


def split_multiple(record, motif):
    motifs = []; static = record.values[3:]
    chrom = str(record['chromosome'])
    start = int(record['start'])
    end = int(record['end'])
    # find the position of the first CG dinucleotide
    sequence = record['sequence']
    motif_pos = sequence.find(motif)
    first_motif_pos = motif_pos
    
    while motif_pos != -1:
        key = [chrom, start + motif_pos - first_motif_pos, start + motif_pos - first_motif_pos]
        motifs.append(np.concatenate([key, static]))
        motif_pos = sequence.find(motif, motif_pos + 1)

    return list(motifs)


def get_filters_and_probs(nanopolish, motif):
    conf_pos = nanopolish[nanopolish['log_lik_ratio'] > 2.0]
    conf_neg = nanopolish[nanopolish['log_lik_ratio'] < -2.0]
    nanopolish = pd.concat([conf_pos, conf_neg])

    nanopolish['prob_meth'] = np.exp(nanopolish['log_lik_ratio']) \
         / (1 + np.exp(nanopolish['log_lik_ratio']))
    nanopolish['prob_unmeth'] = 1 \
         / (1 + np.exp(nanopolish['log_lik_ratio']))

    if motif == 'CG':
        single = nanopolish[nanopolish['num_cpgs'] == 1]
        multiple = nanopolish[nanopolish['num_cpgs'] >= 2]
    else:
        single = nanopolish[nanopolish['num_motifs'] == 1]
        multiple = nanopolish[nanopolish['num_motifs'] >= 2]

    return nanopolish, single, multiple

    
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
    
    df = pd.merge(df, positions, right_on=['chr', 'start'], \
        left_on=['chromosome', 'start'])

    try:
        label = np.zeros(len(df), dtype=int)
        label[np.argwhere(df['status'].values == 'mod')] = 1

        df['methyl_label'] = label
    
    except:
        pass

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
    '-p', '--positions', default='', help='position to filter out'
)
@click.option(
    '-m', '--motif', default='CG', help='motif to obtain'
)
@click.option(
    '-ml', '--methyl_label', default='', 
    help='whether to include the methyl lable as an additional column'
)
@click.option(
    '-o', '--output', required=True, 
    help='Path to save modifications called'
)
def main(nanopolish_output, dict_reads, positions, motif, methyl_label, output):

    nanopolish = pd.read_csv(nanopolish_output, sep='\t')

    nanopolish, single, multiple = get_filters_and_probs(nanopolish, motif)
    
    output_motifs = multiple.apply(split_multiple, motif=motif, axis=1)
    multiple_df = pd.DataFrame(
        np.vstack(output_motifs.tolist()), columns=nanopolish.columns
    )

    df_calling = get_labels(pd.concat([single, multiple_df]))

    if methyl_label:
        df_calling['methyl_label'] = int(methyl_label)

    if positions:
        positions = pd.read_csv(positions, sep='\t')
        df_calling = get_positions_only(df_calling, positions)

    if dict_reads: 
        dict_names = ut.load_obj(dict_reads)  
        dict_names = {v: k for k, v in dict_names.items()}
        aa = df_calling['read_name'] + '.txt'
        bb = [dict_names[x] for x in aa.tolist()]
        df_calling['readnames'] = bb

    import pdb; pdb.set_trace()
    df_calling.to_csv(
        os.path.join(output, 'mods_nanopolish_labels.tsv'), sep='\t', index=None
    )


if __name__ == '__main__':
    main()