#!/usr/bin/env python3

import os
import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from sklearn.metrics import roc_curve, auc, f1_score


def plot_barplot(df, output):
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 5), facecolor='white', gridspec_kw={'height_ratios':[6,1]})

    ax.set_ylim(.79, 1.)  # outliers only
    ax2.set_ylim(0, .12)
    import pdb;pdb.set_trace()
    sns.barplot(x="index", y=0, hue='CpG', data=df, ax=ax, palette=['#1f78b4', '#a6cee3'])
    sns.barplot(x="index", y=0, hue='CpG', data=df, ax=ax2,palette=['#1f78b4', '#a6cee3'])

    # ax.set_xlabel("Filters", fontsize=12)
    ax.set_ylabel("Performance", fontsize=12)
    ax2.set_ylabel("", fontsize=12)
    ax.set_ylabel("", fontsize=12)
    # plt.xticks(rotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax.tick_params(bottom = False)
    ax.tick_params(labelbottom = False)
    
    ax.get_xaxis().set_visible(False)

    # ax.legend(
    #     bbox_to_anchor=(0., 1.2, 1., .102),
    #     handles=custom_lines, loc='upper center', 
    #     facecolor='white', ncol=1, fontsize=8, frameon=False
    # )

    d = .01  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  

    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # d = 0.01
    # ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs) 

    ax2.get_legend().remove()

    plt.tight_layout()
    out_dir = os.path.join(output, 'mixed_single_CpG.pdf')
    plt.savefig(out_dir)
    plt.close()


# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='Separate mixid and single CpG human')
@click.option(
    '-ma', '--mixed_accuracies', required=True,
    help='mixed CpG accuracies'
)
@click.option(
    '-sa', '--single_accuracies', required=True,
    help='single CpG accuracies'
)
@click.option(
    '-mp', '--mixed_preds', required=True,
    help='mixed CpG predictions'
)
@click.option(
    '-sp', '--single_preds', required=True,
    help='single CpG predictions'
)
@click.option(
    '-o', '--output', default='', 
    help='Output file extension'
)
def main(mixed_accuracies, single_accuracies, mixed_preds, single_preds, output):
    mix_acc = pd.read_csv(mixed_accuracies, sep='\t')
    sing_acc = pd.read_csv(single_accuracies, sep='\t')

    mix_preds = pd.read_csv(mixed_preds, sep='\t')
    sing_preds = pd.read_csv(single_preds, sep='\t')

    fpr_mix, tpr_mix, _ = roc_curve(
        mix_preds['labels'].values, mix_preds['probs'].values
    )
    mix_acc['AUC'] = auc(fpr_mix, tpr_mix)
    mix_acc = mix_acc.T.reset_index()
    mix_acc['CpG'] = 'Mixed'
    
    fpr_sing, tpr_sing, _ = roc_curve(
        sing_preds['labels'].values, sing_preds['probs'].values
    )
    sing_acc['AUC'] = auc(fpr_sing, tpr_sing)
    sing_acc = sing_acc.T.reset_index()
    sing_acc['CpG'] = 'Single'

    df = pd.concat([mix_acc, sing_acc]).reset_index(drop=True)

    plot_barplot(df, output)
    # import pdb;pdb.set_trace()
    

if __name__ == '__main__':
    main()