#!/usr/bin/envs python3
import os
import click
import pybedtools
import numpy as np
import pandas as pd

chr_list = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', \
    'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', \
    'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chrX']

chr_dict = {'chr1': 1, 'chr2': 2, 'chr3': 3, 'chr4': 4, 'chr5': 5, 
    'chr6': 6, 'chr7': 7, 'chr8': 8, 'chr9': 9, 'chr10': 10, 'chr11': 11, 
    'chr12': 12, 'chr13': 13, 'chr14': 14, 'chr15': 15, 'chr16': 16, 
    'chr17': 17, 'chr18': 18, 'chr19': 19, 'chr20': 20, 'chr21': 21, 
    'chr22': 22, 'chrX': 23, 'chrY': 24}


def chr2num(df):
    df['chr'] = df.chr.apply(lambda x : chr_dict[x])
    return df


def get_replicate_mods(rep1, rep2):
    df = pd.DataFrame()

    for el in chr_list:

        rep1_mod_chr = rep1[rep1[0] == el]
        rep2_mod_chr = rep2[rep2[0] == el]

        merged = pd.merge(rep1_mod_chr, rep2_mod_chr, on=[1], how='inner')
        
        print(merged.shape)
        df = pd.concat([df, merged])
    # import pdb;pdb.set_trace()
    df = df.drop(columns=['0_y', '2_y', '5_y'])
    df.columns = ['chr', 'start', 'end', 'strand', 'cov_rep1', 
        'mod_rep1', 'cov_rep2', 'mod_rep2']

    df['difference'] = abs(df['mod_rep1'] - df['mod_rep2'])
    df_filt = df[df['difference'] < 10]

    df_filt['average_freq'] = (df_filt['mod_rep1'] + df_filt['mod_rep2']) / 2

    df_filt = chr2num(df_filt)
    import pdb; pdb.set_trace()
    return df_filt


def get_line_regions(regions):

    reg = pd.read_csv(regions, sep='\t', compression='gzip')
    reg = reg[(reg['repClass'] == 'LINE') & (reg['repFamily'] == 'L1')]

    reg = reg[['genoName', 'genoStart', 'genoEnd', 'strand', '#bin', 'swScore', 
        'milliDiv', 'milliDel', 'milliIns', 'repName', 'repClass', 
            'repFamily', 'repStart', 'repEnd', 'repLeft', 'id']]

    reg[[0, 1, 2]] = reg['genoName'].str.split('_', expand=True)
    reg = reg.loc[reg[1].isna()]
    reg.drop([0, 1, 2, '#bin', 'swScore', 'milliDiv', 'milliDel', 'milliIns', 
        'repStart', 'repEnd', 'repLeft', 'id', 'repClass', 'repFamily', 
        'repName'], axis=1, inplace=True)

    reg.columns = ['chr', 'start', 'end', 'strand']
    reg = chr2num(reg)
    import pdb;pdb.set_trace()
    return reg, 'line1'


def get_imprinting_regions(regions):
    
    reg = pd.read_csv(regions, sep='\t', compression='gzip')
    reg.drop(['Gene stable ID', 'Gene stable ID version'], axis=1, inplace=True)
    
    reg[[0, 1, 2, 3, 4]] = reg['Chromosome/scaffold name'].str.split('_', expand=True)
    reg = reg.loc[reg[1].isna()]
    reg.drop([0, 1, 2, 3, 4], axis=1, inplace=True)

    reg.columns = ['chr', 'start', 'end', 'gene_name']
    reg['chr'].replace('X', 23, inplace=True)
    reg['chr'] = reg['chr'].astype(int)

    return reg, 'imprinting'


def find_regions_bisulfite(pos_bis, regions):

    a = pybedtools.BedTool.from_dataframe(pos_bis)
    b = pybedtools.BedTool.from_dataframe(regions)
    result = a.intersect(b, wao = True)

    df = pd.read_csv(result.fn, sep='\t', names=['chr', 'start', 'end', 'strand', 
        'cov_rep1', 'mod_rep1', 'cov_rep2', 'mod_rep2', 'difference', 'average_freq', 
        'chr_y', 'start_y', 'end_y', 'name|strand', 'Overlapped'])

    df_overlap = df[df['Overlapped'] != 0]

    import pdb;pdb.set_trace()
    return df_overlap
    

@click.command(short_help='Extract positions for NA12878 as DeepMod')
@click.option(
    '-r1', '--replicate_1', help='NA12878 replicate one'
)
@click.option(
    '-r2', '--replicate_2', help='NA12878 replicate two'
)
@click.option(
    '-lr', '--line_regions', help='table containing regions of interest'
)
@click.option(
    '-ir', '--imprinting_regions', help='table containing regions of interest'
)
@click.option(
    '-o', '--output', help='output folder'
)
def main(replicate_1, replicate_2, line_regions, imprinting_regions, output):

    if line_regions:
        reg, label = get_line_regions(line_regions)

    elif imprinting_regions:
        reg, label = get_imprinting_regions(imprinting_regions)
    
    else: 
        reg = ''; label = 'all'

    rep1 = pd.read_csv(replicate_1, header=None, sep='\t', nrows=10000000).drop(columns=[3, 4, 6, 7, 8])
    rep2 = pd.read_csv(replicate_2, header=None, sep='\t', nrows=10000000).drop(columns=[3, 4, 6, 7, 8])
    
    pos_bis = get_replicate_mods(rep1, rep2)

    if isinstance(reg, pd.DataFrame):
        pos_bis = find_regions_bisulfite(pos_bis, reg)
    
    import pdb;pdb.set_trace()
    out_file = os.path.join(output, 'positions_bisulfite_{}.tsv').format(label)
    pos_bis.to_csv(out_file, sep='\t', index=None)
   

if __name__ == "__main__":
    main()