#!/usr/bin/envs python3

import os 
import click
import numpy as np 
import pandas as pd 
from tqdm import tqdm 


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


@click.command(short_help='get modifications from nanopolish')
@click.option(
    '-no', '--nanopolish_output', required=True, 
    help='call methylation output from Nanopolish'
)
@click.option(
    '-o', '--output', required=True, 
    help='Path to save modifications called'
)
def main(nanopolish_output, output):

    nanopolish = pd.read_csv(nanopolish_output, sep='\t')

    nanopolish, single_cpgs, multiple_cpgs = get_filters_and_probs(nanopolish)
    
    output_cpgs = multiple_cpgs.apply(split_multiple_cpgs, axis=1)
    multiple_df = pd.DataFrame(
        np.vstack(output_cpgs.tolist()), columns=nanopolish.columns
    )

    final_df_calling = get_labels(pd.concat([single_cpgs, multiple_df]))

    final_df_calling.to_csv(
        os.path.join(output, 'mods_nanopolish.tsv'), sep='\t', index=None
    )


if __name__ == '__main__':
    main()