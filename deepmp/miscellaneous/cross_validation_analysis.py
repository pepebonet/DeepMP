#!/usr/bin/env python3

import os
import click
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

palette_dict = { 
        'DeepMP':'#08519c',
        'DeepSignal':'#f03b20', 
        'DeepMod':'#238443',
        'Nanopolish':'#238443',
        'Guppy': '#fed976',
        'Megalodon': '#e7298a'
}


def build_df(aucs, accs):
    df = pd.DataFrame()

    for i, j in zip(aucs, accs):

        df_aucs = pd.read_csv(i, sep='\t')
        df_acc = pd.read_csv(j, sep='\t')
        df_aucs.columns = ['index', '0', 'Model']
        df = pd.concat([df, df_aucs, df_acc])

    return df


def plot_figure(df, output):

    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')

    # sns.boxplot(
    #     x="index", y='0', hue='Model', data=df, ax=ax, 
    #     hue_order=['DeepMP', 'Megalodon', 'DeepSignal', 'Nanopolish', 'Guppy'], 
    #     palette=['#08519c', '#e7298a', '#f03b20', '#238443', '#fed976'],
    #     linewidth=0.1, showfliers=False
    # )

    # sns.scatterplot(
    #     x="index", y='0', hue='Model', data=df, ax=ax, 
    #     hue_order=['DeepMP', 'Megalodon', 'DeepSignal', 'Nanopolish', 'Guppy'], 
    #     palette=['#08519c', '#e7298a', '#f03b20', '#238443', '#fed976'], x_jitter=True
    # )
    sns.stripplot(
        x="index", y='0', hue='Model', data=df, ax=ax, 
        hue_order=['DeepMP', 'Megalodon', 'DeepSignal', 'Nanopolish', 'Guppy'], 
        palette=['#08519c', '#e7298a', '#f03b20', '#238443', '#fed976'], alpha=0.9, jitter=True
    )
    custom_lines = []
    for el in [('DeepMP', '#08519c'), ('Megalodon', '#e7298a'), \
        ('DeepSignal', '#f03b20'), ('Nanopolish', '#238443'), ('Guppy', '#fed976')]:
        custom_lines.append(
                plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
                mew=0, color=el[1], label=el[0])[0] 
            )

        means = df[df['Model'] == el[0]].groupby('index').mean()
        
        plt.hlines(y=means.loc['AUC'], xmin=-0.15, xmax=0.15, color=el[1], linestyle="dashed")
        plt.hlines(y=means.loc['Accuracy'], xmin=0.85, xmax=1.15, color=el[1], linestyle="dashed")
        plt.hlines(y=means.loc['Precision'], xmin=1.85, xmax=2.15, color=el[1], linestyle="dashed")
        plt.hlines(y=means.loc['Recall'], xmin=2.85, xmax=3.15, color=el[1], linestyle="dashed")
        plt.hlines(y=means.loc['F-score'], xmin=3.85, xmax=4.15, color=el[1], linestyle="dashed")


    ax.set_ylabel("Performance", fontsize=12)
    ax.set_ylabel("", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=2, fontsize=8, frameon=False
    )

    plt.tight_layout()
    out_dir = os.path.join(output, 'cross_validation_5fold.pdf')
    plt.savefig(out_dir)
    plt.close()


@click.command(short_help='SVM accuracy output')
@click.option(
    '-aucs', '--aucs_methods', required=True, multiple=True, 
    help='auc values from different methods'
)
@click.option(
    '-accs', '--accuracies_methods', required=True, multiple=True, 
    help='accuracy values from different methods'
)
@click.option(
    '-o', '--output', default='', help='output folder'
)
def main(aucs_methods, accuracies_methods, output):

    df = build_df(aucs_methods, accuracies_methods)

    plot_figure(df, output)
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()
