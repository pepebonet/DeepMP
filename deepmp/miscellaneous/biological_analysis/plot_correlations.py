#!/usr/bin/envs/ python3

import os
import click
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def get_deepmp_freq(deepmp_output, bisulfite, cov):

    deepmp = pd.read_csv(deepmp_output, sep='\t')

    deepmp_pos = do_per_position_deepmp(deepmp, cov)

    deepmp_freq = pd.merge(
        deepmp_pos, bisulfite, right_on=['chr', 'start', 'strand'], 
        left_on=['chrom', 'pos', 'strand']).drop_duplicates(subset=['id'])
    
    return deepmp_freq['meth_freq']*100, deepmp_freq['average_freq']


def do_per_position_deepmp(df, coverage):

    df['id'] = df['chrom'] + '_' + df['pos'].astype(str) + '_' + df['strand']
    chromosome, pos, strand, cov, ids, meth_freq = [], [], [], [], [], []
    
    for i, j in df.groupby('id'):
        if len(j) >= coverage:
            meth_freq.append(round(j['Prediction'].sum() / j.shape[0], 5))
            cov.append(len(j)); ids.append(i)
            chromosome.append(i.split('_')[0])
            strand.append(i.split('_')[2])
            pos.append(i.split('_')[1])

    preds = pd.DataFrame()
    preds['chrom'] = chromosome
    preds['pos'] = pos
    preds['strand'] = strand
    preds['id'] = ids
    preds['cov'] = cov  
    preds['meth_freq'] = meth_freq
    preds['pos'] = preds['pos'].astype(int)

    return preds


def get_deepsignal_freq(deepsignal_output, bisulfite, cov):

    deepsignal = pd.read_csv(deepsignal_output, sep='\t', header=None)
    
    deepsignal_pos = do_per_position_deepsignal(deepsignal, cov)

    deepsignal_freq = pd.merge(
        deepsignal_pos, bisulfite, right_on=['chr', 'start', 'strand'], 
        left_on=['chrom', 'pos', 'strand']).drop_duplicates(subset=['id'])
    
    return deepsignal_freq['meth_freq']*100, deepsignal_freq['average_freq']


def do_per_position_deepsignal(df, coverage):

    df['id'] = df[0] + '_' + df[1].astype(str) + '_' + df[2]
    chromosome, pos, strand, cov, ids, meth_freq = [], [], [], [], [], []
    
    for i, j in df.groupby('id'):
        if len(j) >= coverage:
            meth_freq.append(round(j[8].sum() / j.shape[0], 5))
            cov.append(len(j)); ids.append(i)
            chromosome.append(i.split('_')[0])
            strand.append(i.split('_')[2])
            pos.append(i.split('_')[1])

    preds = pd.DataFrame()
    preds['chrom'] = chromosome
    preds['pos'] = pos
    preds['strand'] = strand
    preds['id'] = ids
    preds['cov'] = cov  
    preds['meth_freq'] = meth_freq
    preds['pos'] = preds['pos'].astype(int)

    return preds


def get_guppy_freq(guppy_output, bisulfite, cov):

    guppy = pd.read_csv(guppy_output, sep='\t')
    
    guppy_pos = do_per_position_guppy(guppy, cov)

    guppy_freq = pd.merge(
        guppy_pos, bisulfite, right_on=['chr', 'start', 'strand'], 
        left_on=['chrom', 'pos', 'strand']).drop_duplicates(subset=['id'])
    
    return guppy_freq['meth_freq']*100, guppy_freq['average_freq']


def do_per_position_guppy(df, coverage):

    df['id'] = df['#chromosome'] + '_' + df['start'].astype(str) + '_' + df['strand']
    chromosome, pos, strand, cov, ids, meth_freq, av_freq = [], [], [], [], [], [], []

    for i, j in df.groupby('id'):
        if len(j) >= coverage:
            meth_freq.append(round(j['Prediction'].sum() / j.shape[0], 5))
            cov.append(len(j)); ids.append(i)
            chromosome.append(i.split('_')[0])
            strand.append(i.split('_')[2])
            pos.append(i.split('_')[1])
            av_freq.append(j['average_freq'].unique()[0])

    preds = pd.DataFrame()
    preds['chrom'] = chromosome
    preds['pos'] = pos
    preds['strand'] = strand
    preds['id'] = ids
    preds['cov'] = cov  
    preds['meth_freq'] = meth_freq
    preds['av_freq'] = av_freq
    preds['pos'] = preds['pos'].astype(int)

    return preds


def get_nanopolish_freq(nanopolish_output, bisulfite, cov):

    nanopolish = pd.read_csv(nanopolish_output, sep='\t')
    
    nanopolish_pos = do_per_position_nanopolish(nanopolish, cov)

    nanopolish_freq = pd.merge(
        nanopolish_pos, bisulfite, right_on=['chr', 'start', 'strand'], 
        left_on=['chrom', 'pos', 'strand']).drop_duplicates(subset=['id'])
    
    return nanopolish_freq['meth_freq']*100, nanopolish_freq['average_freq']


def do_per_position_nanopolish(df, coverage):

    df['id'] = df['chromosome'] + '_' + df['start'].astype(str) + '_' + df['strand']
    chromosome, pos, strand, cov, ids, meth_freq, av_freq = [], [], [], [], [], [], []

    for i, j in df.groupby('id'):
        if len(j) >= coverage:
            meth_freq.append(round(j['Prediction'].sum() / j.shape[0], 5))
            cov.append(len(j)); ids.append(i)
            chromosome.append(i.split('_')[0])
            strand.append(i.split('_')[2])
            pos.append(i.split('_')[1])
            av_freq.append(j['average_freq'].unique()[0])

    preds = pd.DataFrame()
    preds['chrom'] = chromosome
    preds['pos'] = pos
    preds['strand'] = strand
    preds['id'] = ids
    preds['cov'] = cov  
    preds['meth_freq'] = meth_freq
    preds['av_freq'] = av_freq
    preds['pos'] = preds['pos'].astype(int)

    return preds


def get_megalodon_freq(megalodon_output, bisulfite, cov):

    megalodon = pd.read_csv(megalodon_output, sep='\t')
    
    megalodon_pos = do_per_position_megalodon(megalodon, cov)

    megalodon_freq = pd.merge(
        megalodon_pos, bisulfite, right_on=['chr', 'start', 'strand'], 
        left_on=['chrom', 'pos', 'strand']).drop_duplicates(subset=['id'])
    
    return megalodon_freq['meth_freq']*100, megalodon_freq['average_freq']


def do_per_position_megalodon(df, coverage):

    df['id'] = df['chromosome'] + '_' + df['start'].astype(str) + '_' + df['strand']
    chromosome, pos, strand, cov, ids, meth_freq, av_freq = [], [], [], [], [], [], []

    for i, j in df.groupby('id'):
        if len(j) >= coverage:
            meth_freq.append(round(j['Prediction'].sum() / j.shape[0], 5))
            cov.append(len(j)); ids.append(i)
            chromosome.append(i.split('_')[0])
            strand.append(i.split('_')[2])
            pos.append(i.split('_')[1])
            av_freq.append(j['average_freq'].unique()[0])

    preds = pd.DataFrame()
    preds['chrom'] = chromosome
    preds['pos'] = pos
    preds['strand'] = strand
    preds['id'] = ids
    preds['cov'] = cov  
    preds['meth_freq'] = meth_freq
    preds['av_freq'] = av_freq
    preds['pos'] = preds['pos'].astype(int)

    return preds


def plot_scatter(meth, bis, rms, pear, label, output):

    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    custom_lines = []

    plt.scatter(meth, bis, c="g", alpha=0.5, marker='o')
    # import pdb;pdb.set_trace()

    plt.xlabel('Methylation Frequency {}'.format(label))
    plt.ylabel('Methylation Frequency Bisulfite')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig_out = os.path.join(output, '{}.png'.format(label))
    plt.tight_layout()
    plt.savefig(fig_out)
    plt.close()

# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='SVM accuracy output')
@click.option(
    '-do', '--deepmp_output', default='', 
    help='Output table from deepMP'
)
@click.option(
    '-dso', '--deepsignal_output', default='', 
    help='Output table from deepsignal'
)
@click.option(
    '-no', '--nanopolish_output', default='', 
    help='nanopolish output table'
)
@click.option(
    '-go', '--guppy_output', default='', 
    help='guppy output table'
)
@click.option(
    '-mo', '--megalodon_output', default='', 
    help='megalodon output table'
)
@click.option(
    '-bp', '--bisulfite_positions', default='', 
    help='posiitions and methylation frequency given by bisulfite sequencing'
)
@click.option(
    '-c', '--coverage', default=6, 
    help='coverage threshold to perform position analysis'
)
@click.option(
    '-o', '--output', default='', 
    help='Output file extension'
)
def main(deepmp_output, deepsignal_output, nanopolish_output, guppy_output, 
    megalodon_output, bisulfite_positions, coverage, output):
    
    bisulfite = pd.read_csv(bisulfite_positions, sep='\t')

    if deepmp_output:
        deepmp_freq, bis_deepmp = get_deepmp_freq(deepmp_output, bisulfite, coverage)
        rms_deepmp = mean_squared_error(bis_deepmp, deepmp_freq, squared=False)
        pear_deepmp = stats.pearsonr(deepmp_freq, bis_deepmp)
        plot_scatter(
            deepmp_freq, bis_deepmp, rms_deepmp, pear_deepmp, 'DeepMP', output
        )
        print(rms_deepmp, pear_deepmp)
    
    if guppy_output:
        guppy_freq, bis_guppy = get_guppy_freq(guppy_output, bisulfite, coverage)
        rms_guppy = mean_squared_error(bis_guppy, guppy_freq, squared=False)
        pear_guppy = stats.pearsonr(guppy_freq, bis_guppy)
        plot_scatter(
            guppy_freq, bis_guppy, rms_guppy, pear_guppy, 'Guppy', output
        )
        print(rms_guppy, pear_guppy)

    if nanopolish_output:
        nanopolish_freq, bis_nano = get_nanopolish_freq(
            nanopolish_output, bisulfite, coverage
        )
        rms_nanopolish = mean_squared_error(bis_nano, nanopolish_freq, squared=False)
        pear_nano = stats.pearsonr(nanopolish_freq, bis_nano)
        plot_scatter(
            nanopolish_freq, bis_nano, rms_nanopolish, pear_nano, 'Nanopolish', output
        )
        print(rms_nanopolish, pear_nano)

    if deepsignal_output:
        deepsignal_freq, bis_dsig = get_deepsignal_freq(
            deepsignal_output, bisulfite, coverage
        )
        rms_deepsignal = mean_squared_error(bis_dsig, deepsignal_freq, squared=False)
        pear_dsig = stats.pearsonr(deepsignal_freq, bis_dsig)
        plot_scatter(
            deepsignal_freq, bis_dsig, rms_deepsignal, pear_dsig, 'Deepsignal', output
        )
        print(rms_deepsignal, pear_dsig)

    if megalodon_output:
        megalodon_freq, bis_meg = get_megalodon_freq(
            megalodon_output, bisulfite, coverage
        )
        rms_meg = mean_squared_error(bis_meg, megalodon_freq, squared=False)
        pear_meg = stats.pearsonr(megalodon_freq, bis_meg)
        plot_scatter(
            megalodon_freq, bis_meg, rms_meg, pear_meg, 'Megalodon', output
        )
        print(rms_meg, pear_meg)

    import pdb;pdb.set_trace()
    


if __name__ == '__main__':
    main()
