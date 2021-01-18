#!/usr/bin/env python3

import os
import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from sklearn.metrics import roc_curve, auc, f1_score


def read_accs(mixed_accuracies, single_accuracies):

    mix_acc = pd.read_csv(mixed_accuracies, sep='\t')
    sing_acc = pd.read_csv(single_accuracies, sep='\t')

    return mix_acc, sing_acc


def read_preds(mixed_preds, single_preds):

    mix_preds = pd.read_csv(mixed_preds, sep='\t')
    sing_preds = pd.read_csv(single_preds, sep='\t')

    return mix_preds, sing_preds


def plot_barplot(df, output):
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 5), facecolor='white', gridspec_kw={'height_ratios':[7,1]})

    ax.set_ylim(.85, 1.) 
    ax2.set_ylim(0, .12)

    sns.barplot(x="index", y=0, hue='CpG', data=df, ax=ax, palette=['#08519c', '#f03b20', '#a6cee3', '#fc9272'], \
        hue_order=['DeepMP Mixed', 'DeepSignal Mixed', 'DeepMP Single', 'DeepSignal Single'])
    sns.barplot(x="index", y=0, hue='CpG', data=df, ax=ax2, palette=['#08519c', '#f03b20', '#a6cee3', '#fc9272'], \
        hue_order=['DeepMP Mixed', 'DeepSignal Mixed', 'DeepMP Single', 'DeepSignal Single'])

    custom_lines = []
    for el in [('DeepMP Multiple', '#08519c'), ('DeepMP Single', '#a6cee3'), \
        ('DeepSignal Multiple', '#f03b20'), ('DeepSignal Single', '#fc9272')]:
        custom_lines.append(
                plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
                mew=0, color=el[1], label=el[0])[0] 
            )


    ax.set_ylabel("Performance", fontsize=12)
    ax2.set_ylabel("", fontsize=12)
    ax.set_ylabel("", fontsize=12)
    ax2.set_xlabel("", fontsize=12)
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

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=2, fontsize=8, frameon=False
    )

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


def load_txt(path):
    with open(path, 'r') as f:
        return f.read().split(' ')


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
    '-mad', '--mixed_acc_deepsignal', required=True,
    help='mixed CpG accuracies'
)
@click.option(
    '-sad', '--single_acc_deepsignal', required=True,
    help='single CpG accuracies'
)
@click.option(
    '-mud', '--mixed_auc_deepsignal', required=True,
    help='mixed CpG predictions'
)
@click.option(
    '-sud', '--single_auc_deepsignal', required=True,
    help='single CpG predictions'
)
@click.option(
    '-o', '--output', default='', 
    help='Output file extension'
)
def main(mixed_accuracies, single_accuracies, mixed_preds, single_preds, mixed_acc_deepsignal, \
    single_acc_deepsignal, mixed_auc_deepsignal, single_auc_deepsignal, output):
    
    mix_acc, sing_acc = read_accs(mixed_accuracies, single_accuracies)
    mix_preds, sing_preds = read_preds(mixed_preds, single_preds)

    fpr_mix, tpr_mix, _ = roc_curve(
        mix_preds['labels'].values, mix_preds['probs'].values
    )
    mix_acc['AUC'] = auc(fpr_mix, tpr_mix)
    mix_acc = mix_acc.T.reset_index()
    mix_acc['CpG'] = 'DeepMP Mixed'
    
    fpr_sing, tpr_sing, _ = roc_curve(
        sing_preds['labels'].values, sing_preds['probs'].values
    )
    sing_acc['AUC'] = auc(fpr_sing, tpr_sing)
    sing_acc = sing_acc.T.reset_index()
    sing_acc['CpG'] = 'DeepMP Single'

    df_deepmp = pd.concat([mix_acc, sing_acc]).reset_index(drop=True)

    mix_acc_ds, sing_acc_ds = read_accs(mixed_acc_deepsignal, single_acc_deepsignal)

    mix_acc_ds['AUC'] = np.float(load_txt(mixed_auc_deepsignal)[0])
    mix_acc_ds = mix_acc_ds.T.reset_index()
    mix_acc_ds['CpG'] = 'DeepSignal Mixed'

    sing_acc_ds['AUC'] = np.float(load_txt(single_auc_deepsignal)[0])
    sing_acc_ds = sing_acc_ds.T.reset_index()
    sing_acc_ds['CpG'] = 'DeepSignal Single'

    df_deepsignal = pd.concat([mix_acc_ds, sing_acc_ds]).reset_index(drop=True)

    df = pd.concat([df_deepmp, df_deepsignal]).reset_index(drop=True)
    
    import pdb;pdb.set_trace()
    plot_barplot(df, output)
    import pdb;pdb.set_trace()
    

if __name__ == '__main__':
    main()