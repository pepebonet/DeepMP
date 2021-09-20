
import os
import click
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

palette_dict = { 
        'DeepMP':'#08519c',
        'DeepSignal':'#f03b20', 
        'DeepMod':'#238443',
        'Nanopolish':'#238443',
        'Guppy': '#fed976',
        'Megalodon': '#984ea3ff'
}


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def do_heatmap(imp, line1, output):

    data = np.vstack((line1['Pearson'].values, imp['Pearson'].values)).T
    exp = ['LINE1 Regions', 'Imprinting Genes']
    methods = imp['Method'].tolist()

    fig, ax = plt.subplots()

    im, cbar = heatmap(data, methods, exp, ax=ax,
                   cmap="Blues", cbarlabel="Pearson's Correlation")
    
    texts = annotate_heatmap(im, valfmt="{x:.3f}")
    
    out_fig = os.path.join(output, 'heatmap_bio_regions.pdf')
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()


def do_boxplot_comparisons(df, reg, corr, label, plot_type, output):

    df['Method_Frequency'] = df['Method_Frequency'].astype(int)

    #init the figure
    plt.figure()
    sns.set(style="white")
    custom_lines = []

    #select over both plot types
    if plot_type == 'boxplot':
        plot_fct = sns.boxplot
    else:
        plot_fct = sns.violinplot

    b = plot_fct(
        x='Method_Frequency', y=df['Bisulfite_Frequency'], hue='Method', data=df,
        linewidth=1.5, showfliers=False, 
        hue_order=['DeepMP', 'DeepSignal', 'Nanopolish', 'Guppy']
    )

    count = 0
    for pos in df['Method_Frequency'].unique():
        for me in ['DeepMP', 'DeepSignal', 'Nanopolish', 'Guppy']:
            if plot_type == 'boxplot':
                mybox = b.artists[count]
                mybox.set_facecolor(palette_dict[me])
                # mybox.set_edgecolor('black')
                mybox.set_alpha(0.9)
            
            count += 1

    for me in ['DeepMP', 'DeepSignal', 'Nanopolish', 'Guppy']:
        
        lab = '{} (r = {}, r² = {}, RMSE = {}'.format(
            me, np.round(reg[me][2], 3), 
            np.round(corr[corr['Method'] == me]['Pearson'].values[0], 3), 
            np.round(corr[corr['Method'] == me]['RMSE'].values[0], 3)
        )
        
        custom_lines.append(
            plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
            mew=0, color=palette_dict[me], label=lab)[0] 
        )

    b.spines['top'].set_visible(False)
    b.spines['right'].set_visible(False)
    plt.legend().set_visible(False)

    b.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    b.set_xlabel("Method Methylation", fontsize=12)
    b.set_ylabel("Bisulfite Methylation", fontsize=12)

    out_fig = os.path.join(output, 'comparison_{}.pdf'.format(label))
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()


def do_dotplot_comparisons(df, reg, corr, label, output):

    df['Method_Frequency'] = df['Method_Frequency'].astype(int)

    #init the figure
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    custom_lines = []

    for pos in df['Method_Frequency'].unique():
        for me in ['DeepMP', 'DeepSignal', 'Nanopolish', 'Guppy']:
            
            vals = df[(df['Method_Frequency'] == pos) & (df['Method'] == me)]
            bis_val = vals['Bisulfite_Frequency'].values
            Q1, med, Q3 = np.percentile(bis_val, [25, 50, 75])

            yerr = np.array([[med - Q1], [Q3 - med]])
            # import pdb;pdb.set_trace()

            if me == 'DeepMP':
                p = -3
            if me == 'DeepSignal':
                p = -1
            if me == 'Nanopolish':
                p = 1
            if me == 'Guppy':
                p = 3

            plt.errorbar(pos + p, med, yerr=yerr, 
                marker='o', mfc=palette_dict[me], mec='black', ms=5, 
                mew=0.5,  ecolor='black', capsize=1, capthick=0.5, elinewidth=0.8, 
                c=palette_dict[me]
            )

    for me in ['DeepMP', 'DeepSignal', 'Nanopolish', 'Guppy']:

        x_reg, y_reg, r2, slope, intercept = reg[me]
        y_reg_set = list(dict.fromkeys(y_reg).keys())
        x_reg_set = list(dict.fromkeys(x_reg.flatten()).keys())
        x_reg_set.insert(0, 0); y_reg_set.insert(0, intercept)
        plt.plot(
            x_reg_set, y_reg_set, color=palette_dict[me], linewidth=1, 
            alpha= 0.6, ls='-'             
        )
        
        lab = '{} (r = {}, r² = {}, RMSE = {}'.format(
            me, 
            np.round(corr[corr['Method'] == me]['Pearson'].values[0], 3), 
            np.round(reg[me][2], 3), 
            np.round(corr[corr['Method'] == me]['RMSE'].values[0], 3)
        )
        
        custom_lines.append(
            plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
            mew=0, color=palette_dict[me], label=lab)[0] 
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend().set_visible(False)

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    ax.set_xlabel("Method Methylation", fontsize=12)
    ax.set_ylabel("Bisulfite Methylation", fontsize=12)

    out_fig = os.path.join(output, 'comparison_dotplot_{}.pdf'.format(label))
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()
    


def linear_regression(df):
    
    reg_dict = {}
    from sklearn.linear_model import LinearRegression
    for i, l in df.groupby('Method'):

        y = l['Method_Frequency'].values
        x = l['Bisulfite_Frequency'].values.reshape((-1, 1))
        model = LinearRegression().fit(x, y)
        r_sq = model.score(x, y)
        print(i, r_sq)
        slope = model.coef_ ; intercept = model.intercept_
        # Make predictions
        y_pred = model.predict(x)

        reg_dict.update({i: (x, y_pred, r_sq, slope, intercept)})
    
    return reg_dict


def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats

    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def plot_line_CI(df, reg, corr, label, output):

    z = 1.96
    if label == 'all':
        df['Bisulfite_Frequency'] = df['Bisulfite_Frequency'].map(
            lambda x: int(round(x/5.0)*5.0) 
        )
    else:
        df['Bisulfite_Frequency'] = df['Bisulfite_Frequency'].map(
            lambda x: int(round(x/25.0)*25.0) 
        )
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    custom_lines = []
    
    for i, j in df.groupby('Method'):

        x_reg, y_reg, r2, slope, intercept = reg[i]
        y_reg_set = list(dict.fromkeys(y_reg).keys())
        x_reg_set = list(dict.fromkeys(x_reg.flatten()).keys())
        x_reg_set.insert(0, 0); y_reg_set.insert(0, intercept)
        
        lab = '{} (r = {}, r² = {}, RMSE = {})'.format(
            i, 
            np.round(corr[corr['Method'] == i]['Pearson'].values[0], 3), 
            np.round(reg[i][2], 3), 
            np.round(corr[corr['Method'] == i]['RMSE'].values[0], 3)
        )
        
        dmp = j[j['Method'] == i]
        data = dmp.groupby(['Bisulfite_Frequency'], as_index=False).mean()
        data2= dmp.groupby(['Bisulfite_Frequency'], as_index=False).std()
        data3= dmp.groupby(['Bisulfite_Frequency'], as_index=False).count()
        x = data['Bisulfite_Frequency'].values
        y = data['Method_Frequency'].values
        s = data2['Method_Frequency'].values
        n = data3['Method_Frequency'].values

        ci = z * s/np.sqrt(n)
        plt.plot(x, y, label=i, color=palette_dict[i])
        plt.fill_between(x, (y-ci), (y+ci), alpha=.3, color=palette_dict[i])
        # import pdb;pdb.set_trace()
        custom_lines.append(
            plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
            mew=0, color=palette_dict[i], label=lab)[0] 
        )


    plt.plot(dmp['Bisulfite_Frequency'], dmp['Bisulfite_Frequency'], 
        '--', label='WGBS', color='grey')
    
    custom_lines.append(
        plt.plot([],[], "--", color='grey', label='Identity')[0] 
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend().set_visible(False)

    plt.xlabel('Bisulfite Frequency')
    plt.ylabel('Method Frequency')

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )
    
    out_fig = os.path.join(output, '{}_genome_comparison.pdf'.format(label))
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()


def plot_line_CI_regions(df, reg, corr, label, output):

    z = 1.96

    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    custom_lines = []
    
    for i, j in df.groupby('Method'):

        x_reg, y_reg, r2, slope, intercept = reg[i]
        y_reg_set = list(dict.fromkeys(y_reg).keys())
        x_reg_set = list(dict.fromkeys(x_reg.flatten()).keys())
        x_reg_set.insert(0, 0); y_reg_set.insert(0, intercept)
        
        lab = '{} (r = {}, r² = {}, RMSE = {})'.format(
            i, 
            np.round(corr[corr['Method'] == i]['Pearson'].values[0], 3), 
            np.round(reg[i][2], 3), 
            np.round(corr[corr['Method'] == i]['RMSE'].values[0], 3)
        )
        
        dmp = j[j['Method'] == i]
        data = dmp.groupby(['Method_Frequency'], as_index=False).mean()
        data2= dmp.groupby(['Method_Frequency'], as_index=False).std()
        data3= dmp.groupby(['Method_Frequency'], as_index=False).count()
        x = data['Bisulfite_Frequency'].values
        y = data['Method_Frequency'].values
        s = data2['Bisulfite_Frequency'].values
        n = data3['Bisulfite_Frequency'].values

        ci = z * s/np.sqrt(n)
        plt.plot(x, y, label=i, color=palette_dict[i])
        plt.fill_between(x, (y-ci), (y+ci), alpha=.3, color=palette_dict[i])
        # import pdb;pdb.set_trace()
        custom_lines.append(
            plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
            mew=0, color=palette_dict[i], label=lab)[0] 
        )


    plt.plot(dmp['Bisulfite_Frequency'], dmp['Bisulfite_Frequency'], 
        '--', label='WGBS', color='grey')
    
    custom_lines.append(
        plt.plot([],[], "--", color='grey', label='Identity')[0] 
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend().set_visible(False)

    plt.xlabel('Bisulfite Frequency')
    plt.ylabel('Method Frequency')

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )
    
    out_fig = os.path.join(output, '{}_region_comparison.pdf'.format(label))
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()


def boxplot_comparisons_ranges(df, label, output):

    df1 = df
    df2 = df[['Bisulfite_Frequency']]
    df2['Method'] ='WGBS'
    df2['Method_Frequency'] = df2['Bisulfite_Frequency']
    df = pd.concat([df1,df2])

    fig, axs = plt.subplots(figsize=(15, 5), facecolor='white', nrows=1, ncols=3)
    axs = axs.flatten()

    low = df[df['Bisulfite_Frequency']< 30].sort_values(by='Method')
    sns.boxplot(x='Method', y='Method_Frequency',data=low, ax=axs[0])

    mid = df[df['Bisulfite_Frequency'].between(30, 70)].sort_values(by='Method')
    sns.boxplot(x='Method', y='Method_Frequency',data=mid, ax=axs[1])

    high = df[df['Bisulfite_Frequency']>=70].sort_values(by='Method')
    sns.boxplot(x='Method', y='Method_Frequency',data=high, ax=axs[2])

    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)


    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)


    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    plt.legend().set_visible(False)

    axs[0].set_xlabel('')
    axs[1].set_xlabel('')
    axs[2].set_xlabel('')
    axs[0].set_ylabel('Method Frequency', fontsize=16)
    axs[1].set_ylabel('Method Frequency', fontsize=16)
    axs[2].set_ylabel('Method Frequency', fontsize=16)

    out_fig = os.path.join(output, '{}_boxplot_region_comparison.pdf'.format(label))
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()



# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='SVM accuracy output')
@click.option(
    '-ir', '--imprinting_results', default='', 
    help='Output table from imprinting results'
)
@click.option(
    '-lr', '--line1_results', default='', 
    help='Output table from line1 results'
)
@click.option(
    '-if', '--imprinting_freq', default='', 
    help='Output table from imprinting frequencies'
)
@click.option(
    '-lf', '--line1_freq', default='', 
    help='Output table from line1 frequencies'
)
@click.option(
    '-ar', '--all_results', default='', 
    help='Output table from all genome results'
)
@click.option(
    '-af', '--all_freq', default='', 
    help='Output table from all genome frequencies'
)
@click.option(
    '-o', '--output', default='', 
    help='Output file extension'
)
def main(imprinting_results, line1_results, imprinting_freq, line1_freq,
    all_freq, all_results, output):

    if imprinting_results and imprinting_freq:

        imp_freq = pd.read_csv(imprinting_freq, sep='\t')
        imp = pd.read_csv(imprinting_results, sep='\t')
        reg_imp = linear_regression(imp_freq)
        do_dotplot_comparisons(imp_freq, reg_imp, imp, 'Imprinting Genes', output)
        plot_line_CI(imp_freq, reg_imp, imp, 'Imprinting Genes', output)
        boxplot_comparisons_ranges(imp_freq, 'Imprinting Genes', output)
    
    if line1_results and line1_freq:

        line1_freq = pd.read_csv(line1_freq, sep='\t')
        line1 = pd.read_csv(line1_results, sep='\t')
        reg_line1 = linear_regression(line1_freq)
        do_dotplot_comparisons(line1_freq, reg_line1, line1, 'LINE1', output)
        plot_line_CI(line1_freq, reg_line1, line1, 'LINE1', output)
        boxplot_comparisons_ranges(line1_freq, 'LINE1', output)

    if all_freq: 
        all_freq = pd.read_csv(all_freq, sep='\t')
        all_res = pd.read_csv(all_results, sep='\t')
        reg_all = linear_regression(all_freq)
        plot_line_CI(all_freq, reg_all, all_res, 'all', output)
    
    # do_heatmap(imp, line1, output)
 
    
if __name__ == '__main__':
    main()
    