import os
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support

def get_accuracy(svm):
    right_mod = svm[(svm['label'] == 'mod') & (svm['prediction'] == 'mod')]
    right_unm = svm[(svm['label'] == 'unm') & (svm['prediction'] == 'unm')]
    return (right_mod.shape[0] + right_unm.shape[0]) / svm.shape[0]

def arrange_labels(df):
    df['binary_pred'] = df['label'].apply(lambda x: get_labels(x))
    return df


def get_labels(x):
    if x == 'unm':
        return 0
    else:
        return 1


def plot_ROC (deepsignal, deepmp, original, fig_out, kn='Linear'):
    # fpr_svm, tpr_svm, thresholds = roc_curve(
    #     svm['binary_pred'].values, svm['ProbM'].values
    # )
    fpr_dmp, tpr_dmp, thresholds = roc_curve(
        deepmp['labels'].values, deepmp['probs'].values
    )
    fpr_ds, tpr_ds, thresholds = roc_curve(
        original[11].values, deepsignal[7].values
    )

    # roc_auc_svm = auc(fpr_svm, tpr_svm)
    roc_auc_dmp = auc(fpr_dmp, tpr_dmp)
    roc_auc_ds = auc(fpr_ds, tpr_ds)
    # plt.plot (fpr_svm, tpr_svm, label ='SVM Epinano: {}'.format(round(roc_auc_svm, 3)))
    plt.plot (fpr_dmp, tpr_dmp, label ='DeepMP: {}'.format(round(roc_auc_dmp, 3)))
    plt.plot (fpr_ds, tpr_ds, label ='Deepsignal: {}'.format(round(roc_auc_ds, 3)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Ecoli data')
    plt.legend(loc="lower right")
    plt.savefig(fig_out)


def save_output(acc, output):
    col_names = ['Precision', 'Recall', 'F-score']
    df = pd.DataFrame([acc], columns=col_names)
    df.to_csv(os.path.join(output, 'acc_comparison.txt'), index=False, sep='\t')


# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='SVM accuracy output')
@click.option(
    '-so', '--svm_output', default='', 
    help='Output table from Epinano'
)
@click.option(
    '-do', '--deepmp_output', default='', 
    help='Output table from deepMP'
)
@click.option(
    '-dso', '--deepsignal_output', default='', 
    help='Output table from deepsignal'
)
@click.option(
    '-ot', '--original_test', default='', 
    help='original_test'
)
@click.option(
    '-o', '--output', default='', 
    help='Output file extension'
)
def main(svm_output, deepmp_output, deepsignal_output, original_test, output):

    if svm_output:
        svm = pd.read_csv(svm_output, sep=',')
        accuracy_svm = get_accuracy(svm)
        df_svm = arrange_labels(svm)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            df_svm['binary_pred'].values, np.round(df_svm['ProbM'].values), average='binary'
        )

    if deepmp_output:
        deepmp = pd.read_csv(deepmp_output, sep='\t')

    if deepsignal_output:
        #TODO <JB> Fix sorting
        deepsignal = pd.read_csv(deepsignal_output, sep='\t', header=None)
        original = pd.read_csv(original_test, sep='\t', header=None)
        import pdb;pdb.set_trace()
        # original = original.sort_values(by=[1])
        # deepsignal = deepsignal.sort_values(by=[1])
        precision, recall, f_score, _ = precision_recall_fscore_support(
            original[11].values, deepsignal[8].values, average='binary'
        )

    save_output([precision, recall, f_score], output)

    out_fig = os.path.join(output, 'AUC_comparison.png')
    plot_ROC(deepsignal, deepmp, original, out_fig)

    


if __name__ == "__main__":
    main()