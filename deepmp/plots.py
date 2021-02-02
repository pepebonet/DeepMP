#!/usr/bin/env python3
##### this script needs to debug, call functions with extra care ###########
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
# from deepmp.call_modifications import get_accuracy_pos


def plot_ROC (y_test, probas, fig_out, kn='Linear'):
    fpr, tpr, thresholds = roc_curve(y_test, probas)

    roc_auc = auc(fpr,tpr)
    label = 'Acc model: {}'.format(round(roc_auc, 3), label=label, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Joint model. Epinano data')
    plt.legend(loc="lower right")
    plt.savefig(fig_out)


def plot_distributions(df, output):
    fig, ax = plt.subplots(figsize=(5, 5))

    sns.kdeplot(df['pred_prob'], shade=True)
    fig.tight_layout()
    ax.set_xlim(0,1)
    plt.savefig(os.path.join(output, 'distributions.png'))
    plt.close()


def accuracy_cov(pred, label, cov, output):
    df_dict = {'predictions': pred, 'methyl_label': label, 'Coverage': cov}
    df = pd.DataFrame(df_dict)
    cov = []; acc = []

    for i, j in df.groupby('Coverage'):
        cov.append(i)
        acc.append(get_accuracy_pos(
            j['methyl_label'].tolist(), j['predictions'].tolist())
        )

    fig, ax = plt.subplots(figsize=(5, 5))

    sns.barplot(cov, acc)
    ax.set_ylim(0.92,1)
    fig.tight_layout()

    plt.savefig(os.path.join(output, 'acc_vs_cov.png'))
    plt.close()


def feature_exploration_plots(feat, kmer, output, plot_label, xlim):

    feat.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.displot(x=feat[8].tolist(), hue=feat['calls'].tolist(), data=feat, kind='kde')

    fig.tight_layout()
    plt.xlim(left=xlim[0], right=xlim[1])

    plt.savefig(os.path.join(output, plot_label))
    plt.close()


def do_PCA(df, kmer, output, plot_label):
    features = list(df.columns[0:17])
    x = df.loc[:, features].values
    y = df.loc[:,['calls']].values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, 
        columns = ['PC1', 'PC2'])

    principalDF = pd.concat([principalDf, df[['calls']]], axis = 1)

    fig_out = os.path.join(output, 'PCA_{}'.format(plot_label))
    
    plot_PCA(principalDF, fig_out)


def plot_PCA(finalDf, fig_out):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC1', fontsize = 15)
    ax.set_ylabel('PC2', fontsize = 15)
    ax.set_title('PCA', fontsize = 20)
    targets = ['FN', 'FP', 'TN', 'TP']
    colors = ['red', 'green', 'blue', 'orange']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['calls'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
                , finalDf.loc[indicesToKeep, 'PC2']
                , c = color
                , s = 1)
    ax.legend(targets)

    plt.savefig(fig_out)