#!/usr/bin/env python3 

import os
import sys
import click 
import numpy as np
import pandas as pd 


sys.path.append('../')
import deepmp.utils as ut

def get_positions_only(df, positions):
    import pdb;pdb.set_trace()
    df = pd.merge(
        df, positions, right_on=['chr', 'start', 'strand'], 
        left_on=['chrm', 'pos', 'strand']
    )
    try:
        label = np.zeros(len(df), dtype=int)
        label[np.argwhere(df['status'].values == 'mod')] = 1

        df['methyl_label'] = label

    except: 
        pass

    return df


# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='Script to obtain megalodon output')
@click.option(
    '-mp', '--megalodon_path', required=True, 
    help='Output table from megalodon from sam'
)
@click.option(
    '-dr', '--dict_reads', required=True, 
    help='dict reads'
)
@click.option(
    '-ml', '--methyl_label', default='', 
    help='whether to include the methyl lable as an additional column'
)
@click.option(
    '-p', '--positions', default='', help='position to filter out'
)
@click.option(
    '-o', '--output', default='', help='output path'
)
def main(megalodon_path, dict_reads, methyl_label, positions, output):
    megalodon = pd.read_csv(megalodon_path, sep='\t')

    megalodon['mod_prob'] = np.exp(megalodon['mod_log_prob'])
    megalodon['unmod_prob'] = np.exp(megalodon['can_log_prob'])

    if methyl_label:
        megalodon['methyl_label'] = int(methyl_label)

    if positions:
        positions = pd.read_csv(positions, sep='\t')
        megalodon = get_positions_only(megalodon, positions)
    
    if dict_reads: 
        dict_names = ut.load_obj(dict_reads)  
        dict_names = {v: k for k, v in dict_names.items()}
        aa = megalodon['read_id'] + '.txt'
        bb = [dict_names[x] for x in aa.tolist()]
        megalodon['read_name'] = bb

    megalodon.to_csv(
        os.path.join(output, 'megalodon_results_from_txt.tsv'), sep='\t', 
        index=None
    )
    import pdb;pdb.set_trace()



if __name__ == '__main__':
    main()