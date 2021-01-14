import os
import sys
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support

sys.path.append('../')
import deepmp.utils as ut

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


def plot_ROC_alone(deepmp, fig_out, kn='Linear'):
    fpr_dmp, tpr_dmp, thresholds = roc_curve(
        deepmp['labels'].values, deepmp['probs'].values
    )
    roc_auc_dmp = auc(fpr_dmp, tpr_dmp)
    plt.plot (fpr_dmp, tpr_dmp, lw=2, label ='DeepMP: {}'.format(round(roc_auc_dmp, 3)))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Ecoli data')
    plt.legend(loc="lower right")
    plt.savefig(fig_out)


def plot_ROC_svm(svm, deepmp, fig_out, kn='Linear'):
    fpr_svm, tpr_svm, thresholds = roc_curve(
        svm['binary_pred'].values, svm['ProbM'].values
    )
    fpr_dmp, tpr_dmp, thresholds = roc_curve(
        deepmp['labels'].values, deepmp['probs'].values
    )

    roc_auc_svm = auc(fpr_svm, tpr_svm)
    roc_auc_dmp = auc(fpr_dmp, tpr_dmp)
    plt.plot (fpr_dmp, tpr_dmp, lw=2, label ='DeepMP: {}'.format(round(roc_auc_dmp, 3)))
    plt.plot (fpr_svm, tpr_svm, lw=2, label ='Epinano: {}'.format(round(roc_auc_svm, 3)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Ecoli data')
    plt.legend(loc="lower right")
    plt.savefig(fig_out)


def plot_ROC_deepsignal(deepsignal, deepmp, fig_out, kn='Linear'):
    fpr_dmp, tpr_dmp, thresholds = roc_curve(
        deepmp['labels'].values, deepmp['probs'].values
    )
    fpr_ds, tpr_ds, thresholds = roc_curve(
        deepsignal[11].values, deepsignal['7_x'].values
    )

    roc_auc_dmp = auc(fpr_dmp, tpr_dmp)
    roc_auc_ds = auc(fpr_ds, tpr_ds)
    plt.plot (fpr_dmp, tpr_dmp, lw=2, label ='DeepMP: {}'.format(round(roc_auc_dmp, 3)))
    plt.plot (fpr_ds, tpr_ds, label ='Deepsignal: {}'.format(round(roc_auc_ds, 3)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Ecoli data')
    plt.legend(loc="lower right")
    plt.savefig(fig_out)


def plot_ROC_all(deepmp, deepsignal, deepmod, fig_out, kn='Linear'):

    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    custom_lines = []

    fpr_dmp, tpr_dmp, thresholds = roc_curve(
        deepmp['labels'].values, deepmp['probs'].values
    )
    fpr_ds, tpr_ds, thresholds = roc_curve(
        deepsignal[11].values, deepsignal['7_x'].values
    )
    fpr_dmo, tpr_dmo, thresholds = roc_curve(
        deepmod['labels'].values, deepmod['probs'].values
    )

    roc_auc_dmp = auc(fpr_dmp, tpr_dmp)
    roc_auc_ds = auc(fpr_ds, tpr_ds)
    roc_auc_dmo = auc(fpr_dmo, tpr_dmo)

    # plt.plot (fpr_dmp, tpr_dmp, lw=2, label ='DeepMP: {}'.format(round(roc_auc_dmp, 3)), c='#08519c')
    plt.plot (fpr_dmp, tpr_dmp, lw=2, c='#08519c')
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=1, color='#08519c', label='DeepMP AUC: {}'.format(round(roc_auc_dmp, 3)))[0] 
    )
    # plt.plot (fpr_ds, tpr_ds, lw=2, label ='Deepsignal: {}'.format(round(roc_auc_ds, 3)), c='#f03b20')
    plt.plot (fpr_ds, tpr_ds, lw=2, c='#f03b20')
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=1, color='#f03b20', label='DeepSignal AUC: {}'.format(round(roc_auc_ds, 3)))[0] 
    )
    # plt.plot (fpr_dmo, tpr_dmo, lw=2, label ='DeepMod: {}'.format(round(roc_auc_dmo, 3)), c='#238443')
    plt.plot (fpr_dmo, tpr_dmo, lw=2, c='#238443')
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=1, color='#238443', label='DeepMod AUC: {}'.format(round(roc_auc_dmo, 3)))[0] 
    )

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.title('Ecoli data')
    # plt.legend(loc="lower right", title='AUC',  fontsize=8, frameon=False, facecolor='white', handles=custom_lines)
    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    plt.tight_layout()
    plt.savefig(fig_out)
    plt.close()


def plot_precision_recall_curve(labels, probs, fig_out):
    lr_precision, lr_recall, _ = precision_recall_curve(labels, probs)
    lr_auc = auc(lr_recall, lr_precision)

    # summarize scores
    print('auc=%.3f' % (lr_auc))
    # plot the precision-recall curves
    plt.plot(lr_recall, lr_precision, marker='.', label='AUC: {}'.format(lr_auc))
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    plt.legend()
    plt.savefig(fig_out)

    plt.close()


def plot_precision_recall_curve_all(deepmp, deepsignal, deepmod, fig_out):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    custom_lines = []

    dmp_prec, dmp_rec, _ = precision_recall_curve(
        deepmp['labels'].values, deepmp['probs'].values
    )
    ds_prec, ds_rec, _ = precision_recall_curve(
        deepsignal[11].values, deepsignal['7_x'].values
    )
    dmo_prec, dmo_rec, _ = precision_recall_curve(
        deepmod['labels'].values, deepmod['probs'].values
    )

    auc_dmp = auc(dmp_rec, dmp_prec)
    auc_ds = auc(ds_rec, ds_prec)
    auc_dmo = auc(dmo_rec, dmo_prec)

    # plot the precision-recall curves
    # plt.plot(dmp_rec, dmp_prec, lw=2, label='DeepMP: {}'.format(round(auc_dmp, 3)), c='#08519c')
    plt.plot(dmp_rec, dmp_prec, lw=2, c='#08519c')
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=1, color='#08519c', label='DeepMP AUC: {}'.format(round(auc_dmp, 3)))[0] 
    )
    # plt.plot(ds_rec, ds_prec, lw=2, label='DeepSignal: {}'.format(round(auc_ds, 3)), c='#f03b20')
    plt.plot(ds_rec, ds_prec, lw=2, c='#f03b20')
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=1, color='#f03b20', label='DeepSignal AUC: {}'.format(round(auc_ds, 3)))[0] 
    )
    # plt.plot(dmo_rec, dmo_prec, lw=2, label='DeepMod: {}'.format(round(auc_dmo, 3)), c='#238443')
    plt.plot(dmo_rec, dmo_prec, lw=2, c='#238443')
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=1, color='#238443', label='DeepMod AUC: {}'.format(round(auc_dmo, 3)))[0] 
    )
    # axis labels
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # plt.legend(title='AUC', frameon=False, facecolor='white',  fontsize=8, handles=custom_lines, loc="lower left")
    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    plt.tight_layout()
    plt.savefig(fig_out)
    plt.close()


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
    '-dsp', '--deepsignal_probs', default='', 
    help='Output table from deepsignal'
)
@click.option(
    '-dmo', '--deepmod_output', default='',
    help='output from deepmod'
)
@click.option(
    '-ot', '--original_test', default='', 
    help='original_test'
)
@click.option(
    '-o', '--output', default='', 
    help='Output file extension'
)
def main(svm_output, deepmp_output, deepsignal_output, deepsignal_probs, deepmod_output, original_test, output):
    out_fig = os.path.join(output, 'AUC_comparison.pdf')
    if deepmod_output:
        deepmod = pd.read_csv(deepmod_output, sep='\t')

    if deepmp_output:
        deepmp = pd.read_csv(deepmp_output, sep='\t')
        # out_prere = os.path.join(output, 'AUC_prec_recall.png')
        # plot_precision_recall_curve(deepmp['labels'].values, deepmp['probs'].values, out_prere)
        # out_fig_deepmp = os.path.join(output, 'AUC_comparison_deepmp.pdf')
        # plot_ROC_alone(deepmp, out_fig_deepmp)

    if svm_output:
        svm = pd.read_csv(svm_output, sep=',')
        accuracy_svm = get_accuracy(svm)
        df_svm = arrange_labels(svm)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            df_svm['binary_pred'].values, np.round(df_svm['ProbM'].values), average='binary'
        )
        plot_ROC_svm(svm, deepmp, out_fig)
    if deepsignal_output:
        deepsignal = pd.read_csv(deepsignal_output, sep='\t', header=None).drop_duplicates()
        original = pd.read_csv(original_test, sep='\t', header=None).drop_duplicates()

        original['id'] =  original[0] + '_' + original[1].astype(str) + '_' + original[2] \
            + '_' + original[3].astype(str) + '_' + original[4]
        deepsignal['id'] = deepsignal[0] + '_' + deepsignal[1].astype(str) + '_' + deepsignal[2] \
            + '_' + deepsignal[3].astype(str) + '_'  + deepsignal[4]
        merge = pd.merge(deepsignal, original, on='id', how='inner') 
        precision, recall, f_score, _ = precision_recall_fscore_support(
            merge[11].values, merge['8_x'].values, average='binary'
        )
        # import pdb; pdb.set_trace()
        # ut.save_probs(np.maximum(merge['6_x'].values, merge['7_x'].values), merge[11].values, output)
        # out_prere = os.path.join(output, 'AUC_prec_recall.png')
        # import pdb;pdb.set_trace()
        # plot_precision_recall_curve(merge[11].values, merge['7_x'].values, out_prere)
        # plot_ROC_deepsignal(merge, deepmp, out_fig)
    
    # save_output([precision, recall, f_score], output) 
    if deepmod_output:
        fig_out = os.path.join(output, 'comparison_all.pdf')
        plot_ROC_all(deepmp, merge, deepmod, fig_out)   
        out_prere = os.path.join(output, 'AUC_prec_recall_all.pdf')
        plot_precision_recall_curve_all(deepmp, merge, deepmod, out_prere)


if __name__ == "__main__":
    main()