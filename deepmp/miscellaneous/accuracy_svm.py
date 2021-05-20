import os
import sys
import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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


def get_barplot(deepmp_accuracies, merge, precision, recall, f_score, output):
    deepmp_acc = pd.read_csv(deepmp_accuracies, sep='\t')
    deepmp_acc = deepmp_acc.T.reset_index()
    deepmp_acc['Model'] = 'DeepMP'

    test_acc = round(1 - np.argwhere(merge[11].values != merge['8_x'].values).shape[0] / len(merge[11].values), 5)

    deepsignal_acc = pd.DataFrame([[test_acc, precision, recall, f_score]], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    deepsignal_acc = deepsignal_acc.T.reset_index()
    deepsignal_acc['Model'] = 'DeepSignal'
    
    df_acc = pd.concat([deepmp_acc, deepsignal_acc])
    
    plot_barplot(df_acc, output)


def get_accuracies_all(deepmp_accuracies, deepmod_accuracies, merge, precision, recall, f_score, output):
    deepmp_acc = pd.read_csv(deepmp_accuracies, sep='\t')
    deepmp_acc = deepmp_acc.T.reset_index()
    deepmp_acc['Model'] = 'DeepMP'

    test_acc = round(1 - np.argwhere(merge[11].values != merge['8_x'].values).shape[0] / len(merge[11].values), 5)

    deepsignal_acc = pd.DataFrame([[test_acc, precision, recall, f_score]], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    deepsignal_acc = deepsignal_acc.T.reset_index()
    deepsignal_acc['Model'] = 'DeepSignal'

    deepmod_acc = pd.read_csv(deepmod_accuracies, sep='\t')
    deepmod_acc = deepmod_acc.T.reset_index()
    deepmod_acc['Model'] = 'DeepMod'

    df_acc = pd.concat([deepmp_acc, deepsignal_acc, deepmod_acc])

    plot_barplot_all(df_acc, output)


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


def plot_ROC_deepmp(deepmp, deepmp_seq, fig_out, kn='Linear'):

    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    custom_lines = []

    fpr_dmp, tpr_dmp, thresholds = roc_curve(
        deepmp['labels'].values, deepmp['probs'].values
    )
    fpr_ds, tpr_ds, thresholds = roc_curve(
        deepmp_seq['labels'].values, deepmp_seq['probs'].values
    )

    roc_auc_dmp = auc(fpr_dmp, tpr_dmp)
    roc_auc_ds = auc(fpr_ds, tpr_ds)
    
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#08519c', label='Deepmp Seq AUC: {}'.format(round(roc_auc_ds, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#a6cee3', label='DeepMP Seq + Err AUC: {}'.format(round(roc_auc_dmp, 3)))[0] 
    )
    

    plt.plot (fpr_ds, tpr_ds, lw=2, c='#08519c')
    plt.plot (fpr_dmp, tpr_dmp, lw=2, c='#a6cee3')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    plt.tight_layout()
    plt.savefig(fig_out)
    plt.close()


def plot_ROC_deepsignal(deepsignal, deepmp, fig_out, kn='Linear'):

    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    custom_lines = []

    fpr_dmp, tpr_dmp, thresholds = roc_curve(
        deepmp['labels'].values, deepmp['probs'].values
    )
    fpr_ds, tpr_ds, thresholds = roc_curve(
        deepsignal[11].values, deepsignal['7_x'].values
    )

    roc_auc_dmp = auc(fpr_dmp, tpr_dmp)
    roc_auc_ds = auc(fpr_ds, tpr_ds)
    import pdb;pdb.set_trace()
    # with open(os.path.join('../deepsignal/outputs/human/norwich/chr1_analysis/mixed/', 'auc.txt'), 'w') as f: f.write(str(roc_auc_ds))
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#08519c', label='DeepMP AUC: {}'.format(round(roc_auc_dmp, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#f03b20', label='DeepSignal AUC: {}'.format(round(roc_auc_ds, 3)))[0] 
    )

    plt.plot (fpr_ds, tpr_ds, lw=2, c='#f03b20')
    plt.plot (fpr_dmp, tpr_dmp, lw=2, c='#08519c')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    plt.tight_layout()
    plt.savefig(fig_out)
    plt.close()


def plot_ROC_all(deepmp, deepsignal, deepmod, nanopolish, fig_out, kn='Linear'):

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
    fpr_nnp, tpr_nnp, thresholds = roc_curve(
        nanopolish[11].values, nanopolish['prob_meth'].values
    )

    roc_auc_dmp = auc(fpr_dmp, tpr_dmp)
    roc_auc_ds = auc(fpr_ds, tpr_ds)
    roc_auc_dmo = auc(fpr_dmo, tpr_dmo)
    roc_auc_nnp = auc(fpr_nnp, tpr_nnp)

    # plt.plot (fpr_dmp, tpr_dmp, lw=2, label ='DeepMP: {}'.format(round(roc_auc_dmp, 3)), c='#08519c')
    
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#08519c', label='DeepMP AUC: {}'.format(round(roc_auc_dmp, 3)))[0] 
    )
    # plt.plot (fpr_ds, tpr_ds, lw=2, label ='Deepsignal: {}'.format(round(roc_auc_ds, 3)), c='#f03b20')
    
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#f03b20', label='DeepSignal AUC: {}'.format(round(roc_auc_ds, 3)))[0] 
    )
    # plt.plot (fpr_dmo, tpr_dmo, lw=2, label ='DeepMod: {}'.format(round(roc_auc_dmo, 3)), c='#238443')
    
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#238443', label='DeepMod AUC: {}'.format(round(roc_auc_dmo, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#dd1c77', label='Nanopolish AUC: {}'.format(round(roc_auc_nnp, 3)))[0] 
    )

    plt.plot (fpr_dmo, tpr_dmo, lw=2, c='#238443')
    plt.plot (fpr_ds, tpr_ds, lw=2, c='#f03b20')
    plt.plot (fpr_dmp, tpr_dmp, lw=2, c='#08519c')
    plt.plot (fpr_nnp, tpr_nnp, lw=2, c='#dd1c77')

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


def plot_ROC_nanopolish(deepmp, deepsignal, Nanopolish, fig_out, kn='Linear'):

    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    custom_lines = []

    fpr_dmp, tpr_dmp, thresholds = roc_curve(
        deepmp['labels'].values, deepmp['probs'].values
    )
    fpr_ds, tpr_ds, thresholds = roc_curve(
        deepsignal[11].values, deepsignal['7_x'].values
    )
    fpr_dmo, tpr_dmo, thresholds = roc_curve(
        Nanopolish[11].values, Nanopolish['prob_meth'].values
    )

    roc_auc_dmp = auc(fpr_dmp, tpr_dmp)
    roc_auc_ds = auc(fpr_ds, tpr_ds)
    roc_auc_dmo = auc(fpr_dmo, tpr_dmo)

    # plt.plot (fpr_dmp, tpr_dmp, lw=2, label ='DeepMP: {}'.format(round(roc_auc_dmp, 3)), c='#08519c')
    
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#08519c', label='DeepMP AUC: {}'.format(round(roc_auc_dmp, 3)))[0] 
    )
    # plt.plot (fpr_ds, tpr_ds, lw=2, label ='Deepsignal: {}'.format(round(roc_auc_ds, 3)), c='#f03b20')
    
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#f03b20', label='DeepSignal AUC: {}'.format(round(roc_auc_ds, 3)))[0] 
    )
    # plt.plot (fpr_dmo, tpr_dmo, lw=2, label ='Nanopolish: {}'.format(round(roc_auc_dmo, 3)), c='#238443')
    
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#dd1c77', label='Nanopolish AUC: {}'.format(round(roc_auc_dmo, 3)))[0] 
    )

    plt.plot (fpr_dmo, tpr_dmo, lw=2, c='#dd1c77')
    plt.plot (fpr_ds, tpr_ds, lw=2, c='#f03b20')
    plt.plot (fpr_dmp, tpr_dmp, lw=2, c='#08519c')

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


def plot_precision_recall_deepsignal(deepmp, deepsignal, fig_out):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    custom_lines = []

    dmp_prec, dmp_rec, _ = precision_recall_curve(
        deepmp['labels'].values, deepmp['probs'].values
    )
    ds_prec, ds_rec, _ = precision_recall_curve(
        deepsignal[11].values, deepsignal['7_x'].values
    )

    auc_dmp = auc(dmp_rec, dmp_prec)
    auc_ds = auc(ds_rec, ds_prec)

    # plot the precision-recall curves
    
    
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#08519c', label='DeepMP AUC: {}'.format(round(auc_dmp, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#f03b20', label='DeepSignal AUC: {}'.format(round(auc_ds, 3)))[0] 
    )

    plt.plot(ds_rec, ds_prec, lw=2, c='#f03b20')
    plt.plot(dmp_rec, dmp_prec, lw=2, c='#08519c')
    

    # axis labels
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=1, fontsize=8, frameon=False
    )

    plt.tight_layout()
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
    
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#08519c', label='DeepMP AUC: {}'.format(round(auc_dmp, 3)))[0] 
    )
    # plt.plot(ds_rec, ds_prec, lw=2, label='DeepSignal: {}'.format(round(auc_ds, 3)), c='#f03b20')
    
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#f03b20', label='DeepSignal AUC: {}'.format(round(auc_ds, 3)))[0] 
    )
    # plt.plot(dmo_rec, dmo_prec, lw=2, label='DeepMod: {}'.format(round(auc_dmo, 3)), c='#238443')
    
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#238443', label='DeepMod AUC: {}'.format(round(auc_dmo, 3)))[0] 
    )

    plt.plot(dmo_rec, dmo_prec, lw=2, c='#238443')
    plt.plot(ds_rec, ds_prec, lw=2, c='#f03b20')
    plt.plot(dmp_rec, dmp_prec, lw=2, c='#08519c')

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


def plot_precision_recall_curve_nanopolish(deepmp, deepsignal, Nanopolish, fig_out):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    custom_lines = []

    dmp_prec, dmp_rec, _ = precision_recall_curve(
        deepmp['labels'].values, deepmp['probs'].values
    )
    ds_prec, ds_rec, _ = precision_recall_curve(
        deepsignal[11].values, deepsignal['7_x'].values
    )
    dmo_prec, dmo_rec, _ = precision_recall_curve(
        Nanopolish[11].values, Nanopolish['prob_meth'].values
    )

    auc_dmp = auc(dmp_rec, dmp_prec)
    auc_ds = auc(ds_rec, ds_prec)
    auc_dmo = auc(dmo_rec, dmo_prec)

    # plot the precision-recall curves
    # plt.plot(dmp_rec, dmp_prec, lw=2, label='DeepMP: {}'.format(round(auc_dmp, 3)), c='#08519c')
    
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#08519c', label='DeepMP AUC: {}'.format(round(auc_dmp, 3)))[0] 
    )
    # plt.plot(ds_rec, ds_prec, lw=2, label='DeepSignal: {}'.format(round(auc_ds, 3)), c='#f03b20')
    
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#f03b20', label='DeepSignal AUC: {}'.format(round(auc_ds, 3)))[0] 
    )
    # plt.plot(dmo_rec, dmo_prec, lw=2, label='Nanopolish: {}'.format(round(auc_dmo, 3)), c='#238443')
    
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#dd1c77', label='Nanopolish AUC: {}'.format(round(auc_dmo, 3)))[0] 
    )

    plt.plot(dmo_rec, dmo_prec, lw=2, c='#dd1c77')
    plt.plot(ds_rec, ds_prec, lw=2, c='#f03b20')
    plt.plot(dmp_rec, dmp_prec, lw=2, c='#08519c')

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


def plot_barplot(df, output):
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 5), facecolor='white', gridspec_kw={'height_ratios':[7,1]})

    ax.set_ylim(.71, 1.) 
    ax2.set_ylim(0, .12)

    sns.barplot(x="index", y=0, hue='Model', data=df, ax=ax, palette=['#08519c', '#f03b20'])
    sns.barplot(x="index", y=0, hue='Model', data=df, ax=ax2, palette=['#08519c', '#f03b20'])

    custom_lines = []
    for el in [('DeepMP', '#08519c'), ('DeepSignal', '#f03b20')]:
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

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
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

    ax2.get_legend().remove()

    plt.tight_layout()
    out_dir = os.path.join(output, 'accuracies_plot.pdf')
    plt.savefig(out_dir)
    plt.close()



def plot_barplot_all(df, output):
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 5), facecolor='white', gridspec_kw={'height_ratios':[7,1]})

    ax.set_ylim(.71, 1.) 
    ax2.set_ylim(0, .12)

    sns.barplot(x="index", y=0, hue='Model', data=df, ax=ax, palette=['#08519c', '#f03b20', '#238443'])
    sns.barplot(x="index", y=0, hue='Model', data=df, ax=ax2, palette=['#08519c', '#f03b20', '#238443'])

    custom_lines = []
    for el in [('DeepMP', '#08519c'), ('DeepSignal', '#f03b20'), ('DeepMod', '#238443')]:
        custom_lines.append(
                plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
                mew=0, color=el[1], label=el[0])[0] 
            )

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

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    ax.get_xaxis().set_visible(False)

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=3, fontsize=8, frameon=False
    )

    d = .01  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  

    ax2.get_legend().remove()

    plt.tight_layout()
    plt.savefig(output)
    plt.close()



def plot_barplot_deepmp(df, output):
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 5), facecolor='white', gridspec_kw={'height_ratios':[7,1]})

    ax.set_ylim(.48, 1.) 
    ax2.set_ylim(0, .12)

    sns.barplot(x="index", y=0, hue='Model', data=df, ax=ax, palette=['#08519c', '#a6cee3'], 
        hue_order=['DeepMP Seq', 'DeepMP Seq + Err'])
    sns.barplot(x="index", y=0, hue='Model', data=df, ax=ax2, palette=['#08519c', '#a6cee3'], 
        hue_order=['DeepMP Seq', 'DeepMP Seq + Err'])  

    custom_lines = []
    for el in [('DeepMP Seq', '#08519c'), ('DeepSignal Seq + Err', '#a6cee3')]:
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
    out_dir = os.path.join(output, 'accuracies_plot.pdf')
    plt.savefig(out_dir)
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
    '-dos', '--deepmp_output_seq', default='', 
    help='Output table from deepMP sequence module only'
)
@click.option(
    '-da', '--deepmp_accuracies', default='', 
    help='Output accuracies from deepMP'
)
@click.option(
    '-das', '--deepmp_accuracies_seq', default='', 
    help='Output accuracies from deepMP'
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
    '-dmoa', '--deepmod_accuracies', default='',
    help='output accuracies from deepmod'
)
@click.option(
    '-dmo', '--deepmod_output', default='',
    help='output from deepmod'
)
@click.option(
    '-no', '--nanopolish_output', default='', 
    help='nanopolish output table'
)
@click.option(
    '-ot', '--original_test', default='', 
    help='original_test'
)
@click.option(
    '-o', '--output', default='', 
    help='Output file extension'
)
def main(svm_output, deepmp_output, deepmp_output_seq, deepmp_accuracies, 
    deepmp_accuracies_seq, deepsignal_output, deepsignal_probs, deepmod_accuracies, 
    deepmod_output, nanopolish_output, original_test, output):
    out_fig = os.path.join(output, 'AUC_comparison.pdf')

    if deepmod_output:
        deepmod = pd.read_csv(deepmod_output, sep='\t')

    if deepmp_output:
        deepmp = pd.read_csv(deepmp_output, sep='\t')

        if deepmp_output_seq: 
            deepmp_acc = pd.read_csv(deepmp_accuracies, sep='\t')
            deepmp_acc = deepmp_acc.T.reset_index()
            deepmp_acc['Model'] = 'DeepMP Seq + Err'

            deepmp_seq = pd.read_csv(deepmp_output_seq, sep='\t')

            deepmp_acc_seq = pd.read_csv(deepmp_accuracies_seq, sep='\t')
            deepmp_acc_seq = deepmp_acc_seq.T.reset_index()
            deepmp_acc_seq['Model'] = 'DeepMP Seq'

            df = pd.concat([deepmp_acc, deepmp_acc_seq]).reset_index(drop=True)
            plot_barplot_deepmp(df, output)

            fig_deepmp = os.path.join(output, 'deepmp_comparison.pdf')
            plot_ROC_deepmp(deepmp, deepmp_seq, fig_deepmp)
            import pdb;pdb.set_trace()

        out_prere = os.path.join(output, 'AUC_prec_recall.png')
        plot_precision_recall_curve(deepmp['labels'].values, deepmp['probs'].values, out_prere)
        out_fig_deepmp = os.path.join(output, 'AUC_comparison_deepmp.pdf')
        plot_ROC_alone(deepmp, out_fig_deepmp)

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
        
        if deepmp_accuracies:
            get_barplot(deepmp_accuracies, merge, precision, recall, f_score, output)
            
        import pdb;pdb.set_trace()
        # test_acc = round(1 - np.argwhere(merge[11].values != merge['8_x'].values).shape[0] / len(merge[11].values), 5)
        # ut.save_output([test_acc, precision, recall, f_score], output, 'accuracy_measurements.txt')

        roc_fig_ds = os.path.join(output, 'ROC_deepsignal.pdf')
        prc_fig_ds = os.path.join(output, 'PRC_deepsignal.pdf')

        plot_ROC_deepsignal(merge, deepmp, roc_fig_ds)
        plot_precision_recall_deepsignal(deepmp, merge, prc_fig_ds)
    
    if nanopolish_output:
        
        nanopolish = pd.read_csv(nanopolish_output, sep='\t')
        nanopolish['id'] = nanopolish['chromosome'] + '_' + \
            nanopolish['start'].astype(str) + '_+_' + \
                nanopolish['end'].astype(str) + '_' + nanopolish['readnames']
        
        nanopolish_test = pd.merge(nanopolish, original, on='id', how='inner')
        precision, recall, f_score, _ = precision_recall_fscore_support(
            nanopolish_test[11].values, nanopolish_test['Prediction'].values, average='binary'
        )

        nano_acc = round(1 - np.argwhere(nanopolish_test[11].values != \
            nanopolish_test['Prediction'].values).shape[0] / \
                len(nanopolish_test[11].values), 5)

        fig_out = os.path.join(output, 'comparison_nanopolish.pdf')
        plot_ROC_nanopolish(deepmp, merge, nanopolish_test, fig_out)   
        
        out_prere = os.path.join(output, 'AUC_prec_recall_nanopolish.pdf')
        plot_precision_recall_curve_nanopolish(deepmp, merge, nanopolish_test, out_prere)
        
    
    # save_output([precision, recall, f_score], output) 
    if deepmod_output:
        fig_out = os.path.join(output, 'comparison_all.pdf')
        plot_ROC_all(deepmp, merge, deepmod, nanopolish_test, fig_out)   
        import pdb;pdb.set_trace()
        out_prere = os.path.join(output, 'AUC_prec_recall_all.pdf')
        plot_precision_recall_curve_all(deepmp, merge, deepmod, nanopolish_test, out_prere)

        if deepmod_accuracies:
            out_bar_all = os.path.join(output, 'accuracy_comparison_all.pdf')
            get_accuracies_all(deepmp_accuracies, deepmod_accuracies, merge, precision, recall, f_score, out_bar_all)




if __name__ == "__main__":
    main()