#!/usr/bin/envs python3
import os
import click
import numpy as np
import pandas as pd

chr_list = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', \
    'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', \
    'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chrX']

def get_mods(rep1, rep2, status, cov=1):
    df = pd.DataFrame()

    if status == 'mod':
        rep1_mod = rep1[rep1[10] >= 90]
        rep2_mod = rep2[rep2[10] >= 90]
    else:
        rep1_mod = rep1[rep1[10] == 0]
        rep2_mod = rep2[rep2[10] == 0]

    rep1_mod = rep1_mod[rep1_mod[9] >= cov]
    rep2_mod = rep2_mod[rep2_mod[9] >= cov]

    for el in chr_list:
        rep1_mod_chr = rep1_mod[rep1_mod[0] == el]
        rep2_mod_chr = rep2_mod[rep2_mod[0] == el]
        merged = pd.merge(rep1_mod_chr, rep2_mod_chr, on=[1], how='inner')
        print(merged.shape)
        df = pd.concat([df, merged])

    df['status'] = status
    df = df.drop(columns=['0_y', '2_y', '5_y', '9_y', '10_y'])
    df.columns = ['chr', 'start', 'end', 'strand', 'coverage', 'mod_val', 'status']
    
    return df


@click.command(short_help='Extract positions for NA12878 as DeepMod')
@click.option(
    '-r1', '--replicate_1', help='NA12878 replicate one'
)
@click.option(
    '-r2', '--replicate_2', help='NA12878 replicate two'
)
@click.option(
    '-c', '--coverage', default=1, help='coverage threshold'
)
@click.option(
    '-o', '--output', help='output folder'
)
def main(replicate_1, replicate_2, coverage, output):
    rep1 = pd.read_csv(replicate_1, header=None, sep='\t').drop(columns=[3, 4, 6, 7, 8])
    rep2 = pd.read_csv(replicate_2, header=None, sep='\t').drop(columns=[3, 4, 6, 7, 8])

    mod_poss = get_mods(rep1, rep2, 'mod', coverage)
    unmod_poss = get_mods(rep1, rep2, 'unm', coverage)

    positions = pd.concat([mod_poss, unmod_poss])
    print(positions.shape)
    outfile = os.path.join(output, 'NA12878_positions_{}x.tsv'.format(coverage))
    positions.to_csv(outfile, sep='\t', index=None)
    

if __name__ == "__main__":
    main()