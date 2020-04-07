import os
import click
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_modified_sites(df):
    df['kmer'] = df['#Kmer'].apply(lambda x: list(x))
    df1 = pd.DataFrame(df['kmer'].values.tolist(), 
        columns=['base1', 'base2', 'base3', 'base4', 'base5'])
    df2 = df.join(df1) 

    cpg = df2[(df2['base3'] == 'C') & (df2['base2'] == 'G')]
    non_cpg = df2[(df2['base3'] != 'C') | (df2['base2'] != 'G')]
    
    cpg['Status'] = 'mod'
    non_cpg['Status'] = 'unm'

    return pd.concat([cpg, non_cpg])


def clean_df(df):
    return df.drop(columns=['#Kmer', 'Window', 'Ref', 'Coverage',
        'base1', 'base2', 'base3', 'base4', 'base5', 'kmer'])


def do_PCA(df):
    features = list(df.columns[0:5])
    x = df.loc[:, features].values
    y = df.loc[:,['Status']].values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, 
        columns = ['PC1', 'PC2'])

    return pd.concat([principalDf, df[['Status']]], axis = 1)


def plot_PCA(finalDf, fig_out):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC1', fontsize = 15)
    ax.set_ylabel('PC2', fontsize = 15)
    ax.set_title('PCA', fontsize = 20)
    targets = ['mod', 'unm']
    colors = ['red', 'black']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['Status'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
                , finalDf.loc[indicesToKeep, 'PC2']
                , c = color
                , s = 1)
    ax.legend(targets)

    plt.savefig(fig_out)


# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='run pca analysis')
@click.option(
    '-ef', '--error-features', default='', 
    help='error features'
)
@click.option(
    '-t', '--treatment', required=True, 
    type=click.Choice(['treated', 'untreated']),
    help='untreated or treated'
)
@click.option(
    '-o', '--output', default='', help='Output file extension'
)
def main(error_features, output, treatment):
    df = pd.read_csv(error_features)
    all_features = get_modified_sites(df)
    clean_features = clean_df(all_features)

    principalDF = do_PCA(clean_features)
    fig_out = os.path.join(output, 'PCA_{}'.format(treatment))
    plot_PCA(principalDF, fig_out)
    


if __name__ == "__main__":
    main()