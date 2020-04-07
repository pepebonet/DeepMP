import os
import click
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def get_accuracy(svm):
    right_mod = svm[(svm['sample'] == 'mod') & (svm['prediction'] == 'mod')]
    right_unm = svm[(svm['sample'] == 'unm') & (svm['prediction'] == 'unm')]
    return (right_mod.shape[0] + right_unm.shape[0]) / svm.shape[0]

def arrange_labels(df):
    df['binary_pred'] = df['sample'].apply(lambda x: get_labels(x))
    return df


def get_labels(x):
    if x == 'unm':
        return 0
    else:
        return 1


def plot_ROC (y_test, probas, fig_out, kn='Linear'):
    fpr, tpr, thresholds = roc_curve(y_test, probas)

    roc_auc = auc(fpr,tpr)
    plt.plot (fpr, tpr, label = kn + ' kernel (Area under ROC = %0.2f)'% roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiple features (5mC Ecoli)')
    plt.legend(loc="lower right")
    plt.savefig(fig_out)

# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='SVM accuracy output')
@click.option(
    '-so', '--svm_output', default='', 
    help='Output table from Epinano'
)
@click.option(
    '-o', '--output', default='', 
    help='Output file extension'
)
def main(svm_output, output):
    svm = pd.read_csv(svm_output, sep=',')
    
    accuracy = get_accuracy(svm)

    df = arrange_labels(svm)

    out_fig = os.path.join(output, 'AUC.png')
    plot_ROC(df['binary_pred'].values, df['ProbM'].values, out_fig)


if __name__ == "__main__":
    main()