#!/usr/bin/env python3
##### this script needs to debug, call functions with extra care ###########
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from deepmp.call_modifications import get_accuracy_pos

def plot_ROC (y_test, probas, fig_out, kn='Linear'):
    fpr, tpr, thresholds = roc_curve(y_test, probas)

    roc_auc = auc(fpr,tpr)
    label = 'Acc model: {}'.format(round(roc_auc, 3))
    plt.plot (fpr, tpr, label=label, linewidth=2)
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
