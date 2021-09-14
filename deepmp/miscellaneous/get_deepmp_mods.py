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


def get_positions_only(df, positions):
    # import pdb;pdb.set_trace()
    df = pd.merge(
        df, positions, right_on=['chr', 'start', 'strand'], 
        left_on=['#chromosome', 'start', 'strand']
    )
    import pdb;pdb.set_trace()
    try:
        label = np.zeros(len(df), dtype=int)
        label[np.argwhere(df['status'].values == 'mod')] = 1

        df = df[df.columns[:19]]
        df['methyl_label'] = label

    except: 
        pass

    return df


@click.command(short_help='restrict deepmp features to a set of positions')
@click.option(
    '-df', '--deepmp_features', required=True, 
    help='call methylation output from deepmp'
)
@click.option(
    '-p', '--positions', default='', help='position to filter out'
)
@click.option(
    '-o', '--output', required=True, 
    help='Path to save modifications called'
)
def main(deepmp_output, positions, output):

    deepmp = pd.read_csv(deepmp_output, sep='\t')
    
    if positions:
        positions = pd.read_csv(positions, sep='\t')
        deepmp = get_positions_only(deepmp, positions)
    import pdb;pdb.set_trace()

    deepmp.to_csv(
        os.path.join(output, 'mods_deepmp_readnames.tsv'), sep='\t', index=None
    )


if __name__ == '__main__':
    main()