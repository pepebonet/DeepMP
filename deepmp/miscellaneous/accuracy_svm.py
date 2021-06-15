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


def get_barplot(deepmp_accuracies, deepsignal_accs, nanopolish_accs, guppy_accs, 
    megalodon_accs, output):
    deepmp_acc = pd.DataFrame([deepmp_accuracies], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    deepmp_acc = deepmp_acc.T.reset_index()
    deepmp_acc['Model'] = 'DeepMP'

    deepsignal_acc = pd.DataFrame([deepsignal_accs], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    deepsignal_acc = deepsignal_acc.T.reset_index()
    deepsignal_acc['Model'] = 'DeepSignal'

    nanopolish_acc = pd.DataFrame([nanopolish_accs], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    nanopolish_acc = nanopolish_acc.T.reset_index()
    nanopolish_acc['Model'] = 'Nanopolish'

    guppy_acc = pd.DataFrame([guppy_accs], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    guppy_acc = guppy_acc.T.reset_index()
    guppy_acc['Model'] = 'Guppy'

    megalodon_acc = pd.DataFrame([megalodon_accs], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    megalodon_acc = megalodon_acc.T.reset_index()
    megalodon_acc['Model'] = 'Megalodon'
    
    df_acc = pd.concat(
        [deepmp_acc, megalodon_acc, deepsignal_acc, nanopolish_acc, guppy_acc]
    )
    import pdb;pdb.set_trace()
    df_acc.to_csv(os.path.join(output, 'accuracy_measurements_all_methods.tsv'), sep='\t', index=None)
    plot_barplot(df_acc, output)


def get_barplot_pUC19(deepmp_accuracies, deepsignal_accs, nanopolish_accs, 
    megalodon_accs, output):
    deepmp_acc = pd.DataFrame([deepmp_accuracies], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    deepmp_acc = deepmp_acc.T.reset_index()
    deepmp_acc['Model'] = 'DeepMP'

    deepsignal_acc = pd.DataFrame([deepsignal_accs], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    deepsignal_acc = deepsignal_acc.T.reset_index()
    deepsignal_acc['Model'] = 'DeepSignal'

    nanopolish_acc = pd.DataFrame([nanopolish_accs], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    nanopolish_acc = nanopolish_acc.T.reset_index()
    nanopolish_acc['Model'] = 'Nanopolish'

    megalodon_acc = pd.DataFrame([megalodon_accs], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    megalodon_acc = megalodon_acc.T.reset_index()
    megalodon_acc['Model'] = 'Megalodon'
    
    df_acc = pd.concat(
        [deepmp_acc, megalodon_acc, deepsignal_acc, nanopolish_acc]
    )
    # import pdb;pdb.set_trace()
    df_acc.to_csv(os.path.join(output, 'accuracy_measurements_all_methods.tsv'), sep='\t', index=None)
    plot_barplot_pUC19(df_acc, output)


def get_barplot_seq(deepmp_accuracies, deepmp_seq_accuracies, deepsignal_accs, 
    nanopolish_accs, guppy_accs, megalodon_accs, output):
    deepmp_acc = pd.DataFrame([deepmp_accuracies], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    deepmp_acc = deepmp_acc.T.reset_index()
    deepmp_acc['Model'] = 'DeepMP'

    deepmp_seq_acc = pd.DataFrame([deepmp_seq_accuracies], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    deepmp_seq_acc = deepmp_seq_acc.T.reset_index()
    deepmp_seq_acc['Model'] = 'DeepMP Seq'

    deepsignal_acc = pd.DataFrame([deepsignal_accs], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    deepsignal_acc = deepsignal_acc.T.reset_index()
    deepsignal_acc['Model'] = 'DeepSignal'

    nanopolish_acc = pd.DataFrame([nanopolish_accs], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    nanopolish_acc = nanopolish_acc.T.reset_index()
    nanopolish_acc['Model'] = 'Nanopolish'

    guppy_acc = pd.DataFrame([guppy_accs], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    guppy_acc = guppy_acc.T.reset_index()
    guppy_acc['Model'] = 'Guppy'

    megalodon_acc = pd.DataFrame([megalodon_accs], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    megalodon_acc = megalodon_acc.T.reset_index()
    megalodon_acc['Model'] = 'Megalodon'
    
    df_acc = pd.concat(
        [deepmp_acc, deepmp_seq_acc, megalodon_acc, deepsignal_acc, \
            nanopolish_acc, guppy_acc]
    )
    df_acc.to_csv(os.path.join(output, 'accuracy_measurements_all_methods_seq.tsv'), sep='\t', index=None)
    plot_barplot_seq(df_acc, output)


def get_barplot_seq_pUC19(deepmp_accuracies, deepmp_seq_accuracies, deepsignal_accs, 
    nanopolish_accs, megalodon_accs, output):
    deepmp_acc = pd.DataFrame([deepmp_accuracies], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    deepmp_acc = deepmp_acc.T.reset_index()
    deepmp_acc['Model'] = 'DeepMP'

    deepmp_seq_acc = pd.DataFrame([deepmp_seq_accuracies], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    deepmp_seq_acc = deepmp_seq_acc.T.reset_index()
    deepmp_seq_acc['Model'] = 'DeepMP Seq'

    deepsignal_acc = pd.DataFrame([deepsignal_accs], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    deepsignal_acc = deepsignal_acc.T.reset_index()
    deepsignal_acc['Model'] = 'DeepSignal'

    nanopolish_acc = pd.DataFrame([nanopolish_accs], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    nanopolish_acc = nanopolish_acc.T.reset_index()
    nanopolish_acc['Model'] = 'Nanopolish'

    megalodon_acc = pd.DataFrame([megalodon_accs], 
        columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    megalodon_acc = megalodon_acc.T.reset_index()
    megalodon_acc['Model'] = 'Megalodon'
    
    df_acc = pd.concat(
        [deepmp_acc, deepmp_seq_acc, megalodon_acc, deepsignal_acc, \
            nanopolish_acc]
    )
    df_acc.to_csv(os.path.join(output, 'accuracy_measurements_all_methods_seq.tsv'), sep='\t', index=None)
    plot_barplot_seq_pUC19(df_acc, output)


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
        mew=0, color='#08519c', label='Deepmp AUC: {}'.format(round(roc_auc_dmp, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#a6cee3', label='DeepMP Seq AUC: {}'.format(round(roc_auc_ds, 3)))[0] 
    )
    

    plt.plot (fpr_dmp, tpr_dmp, lw=2, c='#08519c')
    plt.plot (fpr_ds, tpr_ds, lw=2, c='#a6cee3')

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


def plot_ROC_guppy(deepmp, deepsignal, Nanopolish, Guppy, fig_out, kn='Linear'):

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
    fpr_guppy, tpr_guppy, thresholds = roc_curve(
        Guppy[11].values, Guppy['prob_meth'].values
    )

    roc_auc_dmp = auc(fpr_dmp, tpr_dmp)
    roc_auc_ds = auc(fpr_ds, tpr_ds)
    roc_auc_dmo = auc(fpr_dmo, tpr_dmo)
    roc_auc_guppy = auc(fpr_guppy, tpr_guppy)

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
        mew=0, color='#fed976', label='Guppy AUC: {}'.format(round(roc_auc_guppy, 3)))[0] 
    )

    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#238443', label='Nanopolish AUC: {}'.format(round(roc_auc_dmo, 3)))[0] 
    )    

    plt.plot (fpr_dmo, tpr_dmo, lw=2, c='#238443')
    plt.plot (fpr_ds, tpr_ds, lw=2, c='#f03b20')
    plt.plot (fpr_dmp, tpr_dmp, lw=2, c='#08519c')
    plt.plot (fpr_guppy, tpr_guppy, lw=2, c='#fed976')

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


def plot_ROC_megalodon(deepmp, deepsignal, Nanopolish, Guppy, megalodon, fig_out, kn='Linear'):

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
    fpr_guppy, tpr_guppy, thresholds = roc_curve(
        Guppy[11].values, Guppy['prob_meth'].values
    )
    fpr_megalodon, tpr_megalodon, thresholds = roc_curve(
        megalodon[11].values, megalodon['7_x'].values
    )

    roc_auc_dmp = auc(fpr_dmp, tpr_dmp)
    roc_auc_ds = auc(fpr_ds, tpr_ds)
    roc_auc_dmo = auc(fpr_dmo, tpr_dmo)
    roc_auc_guppy = auc(fpr_guppy, tpr_guppy)
    roc_auc_megalodon = auc(fpr_megalodon, tpr_megalodon)

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
        mew=0, color='#fed976', label='Guppy AUC: {}'.format(round(roc_auc_guppy, 3)))[0] 
    )

    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#238443', label='Nanopolish AUC: {}'.format(round(roc_auc_dmo, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#e7298a', label='Megalodon AUC: {}'.format(round(roc_auc_megalodon, 3)))[0] 
    )  

    plt.plot (fpr_dmo, tpr_dmo, lw=2, c='#238443')
    plt.plot (fpr_ds, tpr_ds, lw=2, c='#f03b20')
    plt.plot (fpr_dmp, tpr_dmp, lw=2, c='#08519c')
    plt.plot (fpr_guppy, tpr_guppy, lw=2, c='#fed976')
    plt.plot (fpr_megalodon, tpr_megalodon, lw=2, c='#e7298a')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    import pdb;pdb.set_trace()
    aucs_df = pd.DataFrame([['AUC', 'AUC','AUC','AUC','AUC'], \
        [roc_auc_dmp, roc_auc_ds, roc_auc_guppy, roc_auc_dmo, roc_auc_megalodon], \
            ['DeepMP', 'DeepSignal','Guppy','Nanopolish','Megalodon']]).T
            
    aucs_df.to_csv(
        os.path.join(fig_out.rsplit('/', 1)[0], 'aucs_all_methods.tsv'), 
        sep='\t', index=None
    )

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


def plot_ROC_megalodon_pUC19(deepmp, deepsignal, Nanopolish, megalodon, fig_out, kn='Linear'):

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
    fpr_megalodon, tpr_megalodon, thresholds = roc_curve(
        megalodon[11].values, megalodon['7_x'].values
    )

    roc_auc_dmp = auc(fpr_dmp, tpr_dmp)
    roc_auc_ds = auc(fpr_ds, tpr_ds)
    roc_auc_dmo = auc(fpr_dmo, tpr_dmo)
    roc_auc_megalodon = auc(fpr_megalodon, tpr_megalodon)

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

    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#238443', label='Nanopolish AUC: {}'.format(round(roc_auc_dmo, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#e7298a', label='Megalodon AUC: {}'.format(round(roc_auc_megalodon, 3)))[0] 
    )  

    plt.plot (fpr_dmo, tpr_dmo, lw=2, c='#238443')
    plt.plot (fpr_ds, tpr_ds, lw=2, c='#f03b20')
    plt.plot (fpr_dmp, tpr_dmp, lw=2, c='#08519c')
    plt.plot (fpr_megalodon, tpr_megalodon, lw=2, c='#e7298a')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    
    aucs_df = pd.DataFrame([['AUC', 'AUC','AUC','AUC','AUC'], \
        [roc_auc_dmp, roc_auc_ds, roc_auc_dmo, roc_auc_megalodon], \
            ['DeepMP', 'DeepSignal','Guppy','Nanopolish','Megalodon']]).T
            
    aucs_df.to_csv(
        os.path.join(fig_out.rsplit('/', 1)[0], 'aucs_all_methods.tsv'), 
        sep='\t', index=None
    )

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



def plot_ROC_deepmpseq(deepmp, deepmpseq, deepsignal, Nanopolish, Guppy, megalodon, fig_out, kn='Linear'):

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
    fpr_guppy, tpr_guppy, thresholds = roc_curve(
        Guppy[11].values, Guppy['prob_meth'].values
    )
    fpr_megalodon, tpr_megalodon, thresholds = roc_curve(
        megalodon[11].values, megalodon['7_x'].values
    )
    fpr_dmpseq, tpr_dmpseq, thresholds = roc_curve(
        deepmpseq['labels'].values, deepmpseq['probs'].values
    )

    roc_auc_dmp = auc(fpr_dmp, tpr_dmp)
    roc_auc_ds = auc(fpr_ds, tpr_ds)
    roc_auc_dmo = auc(fpr_dmo, tpr_dmo)
    roc_auc_guppy = auc(fpr_guppy, tpr_guppy)
    roc_auc_megalodon = auc(fpr_megalodon, tpr_megalodon)
    roc_auc_dmpseq = auc(fpr_dmpseq, tpr_dmpseq)
    
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#08519c', label='DeepMP AUC: {}'.format(round(roc_auc_dmp, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#e7298a', label='Megalodon AUC: {}'.format(round(roc_auc_megalodon, 3)))[0] 
    )  
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#a6cee3', label='DeepMP Seq AUC: {}'.format(round(roc_auc_dmpseq, 3)))[0] 
    )  
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#f03b20', label='DeepSignal AUC: {}'.format(round(roc_auc_ds, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#fed976', label='Guppy AUC: {}'.format(round(roc_auc_guppy, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#238443', label='Nanopolish AUC: {}'.format(round(roc_auc_dmo, 3)))[0] 
    )
    
    plt.plot (fpr_ds, tpr_ds, lw=2, c='#f03b20')
    plt.plot (fpr_dmp, tpr_dmp, lw=2, c='#08519c')
    plt.plot (fpr_guppy, tpr_guppy, lw=2, c='#fed976')
    plt.plot (fpr_megalodon, tpr_megalodon, lw=2, c='#e7298a')
    plt.plot (fpr_dmpseq, tpr_dmpseq, lw=2, c='#a6cee3')
    plt.plot (fpr_dmo, tpr_dmo, lw=2, c='#238443')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    aucs_df = pd.DataFrame([['AUC', 'AUC', 'AUC','AUC','AUC','AUC'], \
        [roc_auc_dmp, roc_auc_dmpseq, roc_auc_ds, roc_auc_guppy, roc_auc_dmo, roc_auc_megalodon], \
            ['DeepMP', 'DeepMP Seq', 'DeepSignal','Guppy','Nanopolish','Megalodon']]).T
            
    aucs_df.to_csv(
        os.path.join(fig_out.rsplit('/', 1)[0], 'aucs_all_methods_seq.tsv'), 
        sep='\t', index=None
    )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)


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


def plot_ROC_deepmpseq_pUC19(deepmp, deepmpseq, deepsignal, Nanopolish, megalodon, fig_out, kn='Linear'):

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
    fpr_megalodon, tpr_megalodon, thresholds = roc_curve(
        megalodon[11].values, megalodon['7_x'].values
    )
    fpr_dmpseq, tpr_dmpseq, thresholds = roc_curve(
        deepmpseq['labels'].values, deepmpseq['probs'].values
    )

    roc_auc_dmp = auc(fpr_dmp, tpr_dmp)
    roc_auc_ds = auc(fpr_ds, tpr_ds)
    roc_auc_dmo = auc(fpr_dmo, tpr_dmo)
    roc_auc_megalodon = auc(fpr_megalodon, tpr_megalodon)
    roc_auc_dmpseq = auc(fpr_dmpseq, tpr_dmpseq)
    
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#08519c', label='DeepMP AUC: {}'.format(round(roc_auc_dmp, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#e7298a', label='Megalodon AUC: {}'.format(round(roc_auc_megalodon, 3)))[0] 
    )  
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#a6cee3', label='DeepMP Seq AUC: {}'.format(round(roc_auc_dmpseq, 3)))[0] 
    )  
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#f03b20', label='DeepSignal AUC: {}'.format(round(roc_auc_ds, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#238443', label='Nanopolish AUC: {}'.format(round(roc_auc_dmo, 3)))[0] 
    )
    
    plt.plot (fpr_ds, tpr_ds, lw=2, c='#f03b20')
    plt.plot (fpr_dmp, tpr_dmp, lw=2, c='#08519c')
    plt.plot (fpr_megalodon, tpr_megalodon, lw=2, c='#e7298a')
    plt.plot (fpr_dmpseq, tpr_dmpseq, lw=2, c='#a6cee3')
    plt.plot (fpr_dmo, tpr_dmo, lw=2, c='#238443')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)

    aucs_df = pd.DataFrame([['AUC', 'AUC', 'AUC', 'AUC', 'AUC'], \
        [roc_auc_dmp, roc_auc_dmpseq, roc_auc_ds, roc_auc_dmo, roc_auc_megalodon], \
            ['DeepMP', 'DeepMP Seq', 'DeepSignal', 'Nanopolish','Megalodon']]).T
            
    aucs_df.to_csv(
        os.path.join(fig_out.rsplit('/', 1)[0], 'aucs_all_methods_seq.tsv'), 
        sep='\t', index=None
    )

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



def plot_precision_recall_curve_guppy(deepmp, deepsignal, Nanopolish, guppy, fig_out):
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
    guppy_prec, guppy_rec, _ = precision_recall_curve(
        guppy[11].values, guppy['prob_meth'].values
    )

    auc_dmp = auc(dmp_rec, dmp_prec)
    auc_ds = auc(ds_rec, ds_prec)
    auc_dmo = auc(dmo_rec, dmo_prec)
    auc_guppy = auc(guppy_rec, guppy_prec)

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
        mew=0, color='#fed976', label='Guppy AUC: {}'.format(round(auc_guppy, 3)))[0] 
    )

    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#238443', label='Nanopolish AUC: {}'.format(round(auc_dmo, 3)))[0] 
    )

    

    plt.plot(dmo_rec, dmo_prec, lw=2, c='#238443')
    plt.plot(ds_rec, ds_prec, lw=2, c='#f03b20')
    plt.plot(dmp_rec, dmp_prec, lw=2, c='#08519c')
    plt.plot(guppy_rec, guppy_prec, lw=2, c='#fed976')

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


def plot_precision_recall_curve_megalodon(deepmp, deepsignal, Nanopolish, 
    guppy, megalodon, fig_out):
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
    guppy_prec, guppy_rec, _ = precision_recall_curve(
        guppy[11].values, guppy['prob_meth'].values
    )
    megalodon_prec, megalodon_rec, _ = precision_recall_curve(
        megalodon[11].values, megalodon['7_x'].values
    )

    auc_dmp = auc(dmp_rec, dmp_prec)
    auc_ds = auc(ds_rec, ds_prec)
    auc_dmo = auc(dmo_rec, dmo_prec)
    auc_guppy = auc(guppy_rec, guppy_prec)
    auc_megalodon = auc(megalodon_rec, megalodon_prec)

    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#08519c', label='DeepMP AUC: {}'.format(round(auc_dmp, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#238443', label='Megalodon AUC: {}'.format(round(auc_megalodon, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#f03b20', label='DeepSignal AUC: {}'.format(round(auc_ds, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#fed976', label='Guppy AUC: {}'.format(round(auc_guppy, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#238443', label='Nanopolish AUC: {}'.format(round(auc_dmo, 3)))[0] 
    )
    
    
    plt.plot(ds_rec, ds_prec, lw=2, c='#f03b20')
    plt.plot(dmp_rec, dmp_prec, lw=2, c='#08519c')
    plt.plot(guppy_rec, guppy_prec, lw=2, c='#fed976')
    plt.plot(megalodon_rec, megalodon_prec, lw=2, c='#fed976')
    plt.plot(dmo_rec, dmo_prec, lw=2, c='#238443')

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


def plot_precision_recall_curve_deepmpseq(deepmp, deepmpseq, deepsignal, Nanopolish, 
    guppy, megalodon, fig_out):
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
    guppy_prec, guppy_rec, _ = precision_recall_curve(
        guppy[11].values, guppy['prob_meth'].values
    )
    megalodon_prec, megalodon_rec, _ = precision_recall_curve(
        megalodon[11].values, megalodon['7_x'].values
    )
    dmpseq_prec, dmpseq_rec, _ = precision_recall_curve(
        deepmpseq['labels'].values, deepmpseq['probs'].values
    )

    auc_dmp = auc(dmp_rec, dmp_prec)
    auc_ds = auc(ds_rec, ds_prec)
    auc_dmo = auc(dmo_rec, dmo_prec)
    auc_guppy = auc(guppy_rec, guppy_prec)
    auc_megalodon = auc(megalodon_rec, megalodon_prec)
    auc_dmpseq = auc(dmpseq_rec, dmpseq_prec)
    
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#08519c', label='DeepMP AUC: {}'.format(round(auc_dmp, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#e7298a', label='Megalodon AUC: {}'.format(round(auc_megalodon, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#a6cee3', label='DeepMP Seq AUC: {}'.format(round(auc_dmpseq, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#f03b20', label='DeepSignal AUC: {}'.format(round(auc_ds, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#fed976', label='Guppy AUC: {}'.format(round(auc_guppy, 3)))[0] 
    )
    custom_lines.append(
        plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
        mew=0, color='#238443', label='Nanopolish AUC: {}'.format(round(auc_dmo, 3)))[0] 
    )
    
    plt.plot(ds_rec, ds_prec, lw=2, c='#f03b20')
    plt.plot(dmp_rec, dmp_prec, lw=2, c='#08519c')
    plt.plot(guppy_rec, guppy_prec, lw=2, c='#fed976')
    plt.plot(megalodon_rec, megalodon_prec, lw=2, c='#e7298a')
    plt.plot(dmpseq_rec, dmpseq_prec, lw=2, c='#a6cee3')
    plt.plot(dmo_rec, dmo_prec, lw=2, c='#238443')

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


def plot_barplot(df, output):
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 5), facecolor='white', gridspec_kw={'height_ratios':[7,1]})

    ax.set_ylim(.60, 1.) 
    ax2.set_ylim(0, .12)

    sns.barplot(x="index", y=0, hue='Model', data=df, ax=ax, palette=['#08519c', '#e7298a', '#f03b20', '#238443', '#fed976'])
    sns.barplot(x="index", y=0, hue='Model', data=df, ax=ax2, palette=['#08519c', '#e7298a', '#f03b20', '#238443', '#fed976'])

    custom_lines = []
    for el in [('DeepMP', '#08519c'), ('Megalodon', '#e7298a'), ('DeepSignal', '#f03b20'), ('Nanopolish', '#238443'), ('Guppy', '#fed976')]:
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
    out_dir = os.path.join(output, 'accuracies_plot_megalodon.pdf')
    plt.savefig(out_dir)
    plt.close()


def plot_barplot_pUC19(df, output):
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 5), facecolor='white', gridspec_kw={'height_ratios':[7,1]})

    ax.set_ylim(.60, 1.) 
    ax2.set_ylim(0, .12)

    sns.barplot(x="index", y=0, hue='Model', data=df, ax=ax, palette=['#08519c', '#e7298a', '#f03b20', '#238443'])
    sns.barplot(x="index", y=0, hue='Model', data=df, ax=ax2, palette=['#08519c', '#e7298a', '#f03b20', '#238443'])

    custom_lines = []
    for el in [('DeepMP', '#08519c'), ('Megalodon', '#e7298a'), ('DeepSignal', '#f03b20'), ('Nanopolish', '#238443')]:
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
    out_dir = os.path.join(output, 'accuracies_plot_megalodon.pdf')
    plt.savefig(out_dir)
    plt.close()


def plot_barplot_seq(df, output):
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 5), facecolor='white', gridspec_kw={'height_ratios':[7,1]})

    ax.set_ylim(.60, 1.) 
    ax2.set_ylim(0, .12)

    sns.barplot(x="index", y=0, hue='Model', data=df, ax=ax, palette=['#08519c', '#a6cee3', '#e7298a', '#f03b20', '#238443', '#fed976'])
    sns.barplot(x="index", y=0, hue='Model', data=df, ax=ax2, palette=['#08519c', '#a6cee3', '#e7298a', '#f03b20', '#238443', '#fed976'])

    custom_lines = []
    for el in [('DeepMP', '#08519c'), ('DeepMP Seq', '#a6cee3'), ('Megalodon', '#e7298a'), ('DeepSignal', '#f03b20'), ('Nanopolish', '#238443'), ('Guppy', '#fed976')]:
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
    out_dir = os.path.join(output, 'accuracies_plot_deepmp_seq.pdf')
    plt.savefig(out_dir)
    plt.close()


def plot_barplot_seq_pUC19(df, output):
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 5), facecolor='white', gridspec_kw={'height_ratios':[7,1]})

    ax.set_ylim(.60, 1.) 
    ax2.set_ylim(0, .12)

    sns.barplot(x="index", y=0, hue='Model', data=df, ax=ax, palette=['#08519c', '#a6cee3', '#e7298a', '#f03b20', '#238443'])
    sns.barplot(x="index", y=0, hue='Model', data=df, ax=ax2, palette=['#08519c', '#a6cee3', '#e7298a', '#f03b20', '#238443'])

    custom_lines = []
    for el in [('DeepMP', '#08519c'), ('DeepMP Seq', '#a6cee3'), ('Megalodon', '#e7298a'), ('DeepSignal', '#f03b20'), ('Nanopolish', '#238443')]:
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
    out_dir = os.path.join(output, 'accuracies_plot_deepmp_seq_pUC19.pdf')
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
        hue_order=['DeepMP', 'DeepMP Seq'])
    sns.barplot(x="index", y=0, hue='Model', data=df, ax=ax2, palette=['#08519c', '#a6cee3'], 
        hue_order=['DeepMP', 'DeepMP Seq'])  

    custom_lines = []
    for el in [('DeepMP', '#08519c'), ('DeepMP Seq', '#a6cee3')]:
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
    '-go', '--guppy_output', default='', 
    help='guppy output table'
)
@click.option(
    '-mo', '--megalodon_output', default='', 
    help='megalodon output table'
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
    deepmod_output, nanopolish_output, guppy_output, megalodon_output, original_test, output):
    out_fig = os.path.join(output, 'AUC_comparison.pdf')

    if deepmod_output:
        deepmod = pd.read_csv(deepmod_output, sep='\t')
    if deepmp_output:
        deepmp = pd.read_csv(deepmp_output, sep='\t')

        deepmp['id'] = deepmp['chrom'] + '_' + deepmp['pos'].astype(str) + '_' + deepmp['strand'] \
            + '_' + deepmp['pos'].astype(str) + '_' + deepmp['readname']
        # deepmp_min = deepmp[deepmp['probs'] < 0.2]
        # deepmp_max = deepmp[deepmp['probs'] > 0.8]
        # deepmp = pd.concat([deepmp_min, deepmp_max])

        deepmp['Prediction'] = deepmp['probs'].apply(lambda x: 1 if x > 0.5 else 0)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            deepmp['labels'].values, deepmp['Prediction'].values, average='binary'
        )
        deepmp_acc = round(1 - np.argwhere(deepmp['labels'].values != \
            deepmp['Prediction'].values).shape[0] / \
                len(deepmp['labels'].values), 5)
        import pdb;pdb.set_trace()
        accs_deepmp = [deepmp_acc, precision, recall, f_score]
        print(accs_deepmp)
        if deepmp_output_seq: 
            deepmp_acc = pd.read_csv(deepmp_accuracies, sep='\t')
            deepmp_acc = deepmp_acc.T.reset_index()
            deepmp_acc['Model'] = 'DeepMP'

            deepmp_seq = pd.read_csv(deepmp_output_seq, sep='\t')

            deepmp_seq['id'] = deepmp_seq['chrom'] + '_' \
                + deepmp_seq['pos'].astype(str) + '_' + deepmp_seq['strand'] \
                    + '_' + deepmp['pos'].astype(str) + '_' + deepmp['readname']
            # deepmp_seq_min = deepmp_seq[deepmp_seq['probs'] < 0.4]
            # deepmp_seq_max = deepmp_seq[deepmp_seq['probs'] > 0.6]
            # deepmp_seq = pd.concat([deepmp_seq_min, deepmp_seq_max])

            # deepmp_seq['Prediction'] = deepmp_seq['probs'].apply(lambda x: 1 if x > 0.5 else 0)
            precision, recall, f_score, _ = precision_recall_fscore_support(
                deepmp_seq['labels'].values, deepmp_seq['Prediction'].values, average='binary'
            )
            deepmp_seq_acc = round(1 - np.argwhere(deepmp_seq['labels'].values != \
                deepmp_seq['Prediction'].values).shape[0] / \
                    len(deepmp_seq['labels'].values), 5)

            accs_deepmp_seq = [deepmp_seq_acc, precision, recall, f_score]

            deepmp_acc_seq = pd.read_csv(deepmp_accuracies_seq, sep='\t')
            deepmp_acc_seq = deepmp_acc_seq.T.reset_index()
            deepmp_acc_seq['Model'] = 'DeepMP Seq'

            df = pd.concat([deepmp_acc, deepmp_acc_seq]).reset_index(drop=True)
            plot_barplot_deepmp(df, output)

            fig_deepmp = os.path.join(output, 'deepmp_comparison.pdf')
            plot_ROC_deepmp(deepmp, deepmp_seq, fig_deepmp)

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
            + '_' + original[1].astype(str) + '_' + original[4]
        deepsignal['id'] = deepsignal[0] + '_' + deepsignal[1].astype(str) + '_' + deepsignal[2] \
            + '_' + deepsignal[1].astype(str) + '_'  + deepsignal[4]
        # merge = pd.merge(deepsignal, original, on='id', how='inner') 

        # import pdb;pdb.set_trace()
        assert deepsignal.shape[0] == deepmp.shape[0]
        deepsignal = deepsignal.sort_values(by=[1, 4]).reset_index(drop=True)
        deepmp = deepmp.sort_values(by=['pos', 'readname']).reset_index(drop=True)
        deepsignal[11] = deepmp['labels']
        merge = deepsignal.copy()
        merge['7_x'] = merge[7]
        merge['8_x'] = merge[8]

        #TODO delete
        # merge_max = merge[merge['7_x'] > 0.55]
        # merge_min = merge[merge['7_x'] < 0.45]
        # merge = pd.concat([merge_min, merge_max]) 
        # import pdb;pdb.set_trace()
        precision, recall, f_score, _ = precision_recall_fscore_support(
            merge[11].values, merge['8_x'].values, average='binary'
        )

        deepsignal_acc = round(1 - np.argwhere(merge[11].values != merge['8_x'].values).shape[0] / len(merge[11].values), 5)
        accs_deepsignal = [deepsignal_acc, precision, recall, f_score]
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
        
        if nanopolish.shape[0] < 100000:
            nanopolish_test = nanopolish.copy()
            nanopolish_test[11] = nanopolish['methyl_label']
        else:
            nanopolish_test = pd.merge(nanopolish, original, on='id', how='inner')

        precision, recall, f_score, _ = precision_recall_fscore_support(
            nanopolish_test[11].values, nanopolish_test['Prediction'].values, average='binary'
        )
        
        nano_acc = round(1 - np.argwhere(nanopolish_test[11].values != \
            nanopolish_test['Prediction'].values).shape[0] / \
                len(nanopolish_test[11].values), 5)

        accs_nanopolish = [nano_acc, precision, recall, f_score]

        fig_out = os.path.join(output, 'comparison_nanopolish.pdf')
        plot_ROC_nanopolish(deepmp, merge, nanopolish_test, fig_out)  
        
        out_prere = os.path.join(output, 'AUC_prec_recall_nanopolish.pdf')
        plot_precision_recall_curve_nanopolish(deepmp, merge, nanopolish_test, out_prere)

    
    if guppy_output:
        
        guppy = pd.read_csv(guppy_output, sep='\t')
        guppy['id'] = guppy['#chromosome'] + '_' + \
            guppy['start'].astype(str) + '_' + guppy['strand'] + '_' + \
                guppy['start'].astype(str) + '_' + guppy['readnames']
        
        guppy_test = pd.merge(guppy, original, on='id', how='inner')
        precision, recall, f_score, _ = precision_recall_fscore_support(
            guppy_test[11].values, guppy_test['Prediction'].values, average='binary'
        )

        guppy_acc = round(1 - np.argwhere(guppy_test[11].values != \
            guppy_test['Prediction'].values).shape[0] / \
                len(guppy_test[11].values), 5)
        
        accs_guppy = [guppy_acc, precision, recall, f_score]
        import pdb;pdb.set_trace()
        fig_out = os.path.join(output, 'comparison_guppy.pdf')
        plot_ROC_guppy(deepmp, merge, nanopolish_test, guppy_test, fig_out)   
        
        out_prere = os.path.join(output, 'AUC_prec_recall_guppy.pdf')
        plot_precision_recall_curve_guppy(deepmp, merge, nanopolish_test, guppy_test, out_prere)
        

    if megalodon_output:
        import pdb;pdb.set_trace()
        megalodon = pd.read_csv(megalodon_output, sep='\t', header=None)
        # meg_pos = megalodon[megalodon[7] > 0.8]
        # meg_neg = megalodon[megalodon[7] < 0.2]
        # megalodon = pd.concat([meg_pos, meg_neg])
        
        megalodon['Prediction']  = megalodon[7].apply(lambda x: 1 if x > 0.5 else 0)
        original = pd.read_csv(original_test, sep='\t', header=None).drop_duplicates()
        original['id'] =  original[0] + '_' + (original[1]).astype(str) + '_' + \
            original[2] + '_' + (original[1]).astype(str) + '_' + original[4]
        
        megalodon['id'] = megalodon[1] + '_' + megalodon[3].astype(str) + '_' + \
            megalodon[2] + '_' + megalodon[3].astype(str) + '_' + megalodon[9]

        megalodon_test = pd.merge(megalodon, original, on='id', how='inner')

        precision, recall, f_score, _ = precision_recall_fscore_support(
            megalodon_test[11].values, megalodon_test['Prediction'].values, average='binary'
        )

        megalodon_acc = round(1 - np.argwhere(megalodon_test[11].values != \
            megalodon_test['Prediction'].values).shape[0] / \
                len(megalodon_test[11].values), 5)
        
        accs_megalodon = [megalodon_acc, precision, recall, f_score]
        
        deepmp = pd.merge(megalodon_test, deepmp, on='id', how='inner')
        precision, recall, f_score, _ = precision_recall_fscore_support(
            deepmp['labels'].values, deepmp['Prediction_y'].values, average='binary'
        )
        deepmp_acc = round(1 - np.argwhere(deepmp['labels'].values != \
            deepmp['Prediction_y'].values).shape[0] / len(deepmp['labels'].values), 5)
        accs_deepmp = [deepmp_acc, precision, recall, f_score]

        merge = pd.merge(megalodon_test, merge, on='id', how='inner')
        merge['8_x'] = merge['8_x_y']
        merge['7_x'] = merge['7_x_y']
        merge[11] = merge['11_y']
        
        precision, recall, f_score, _ = precision_recall_fscore_support(
            merge[11].values, merge['8_x'].values, average='binary')
        deepsignal_acc = round(1 - np.argwhere(merge[11].values \
            != merge['8_x'].values).shape[0] / len(merge[11].values), 5)
        
        accs_deepsignal = [deepsignal_acc, precision, recall, f_score]

        import pdb;pdb.set_trace()
        if deepmp_accuracies:
            try:
                get_barplot(
                    accs_deepmp, accs_deepsignal, 
                    accs_nanopolish, accs_guppy, accs_megalodon, output
                )
            except: 
                get_barplot_pUC19(
                    accs_deepmp, accs_deepsignal, 
                    accs_nanopolish, accs_megalodon, output
                )
        
        if 'accs_deepmp_seq' in locals():

            deepmp_seq = pd.merge(megalodon_test, deepmp_seq, on='id', how='inner')
            precision, recall, f_score, _ = precision_recall_fscore_support(
                deepmp_seq['labels'].values, deepmp_seq['Prediction_y'].values, average='binary'
            )
            deepmp_seq_acc = round(1 - np.argwhere(deepmp_seq['labels'].values != \
                deepmp_seq['Prediction_y'].values).shape[0] / len(deepmp_seq['labels'].values), 5)
            accs_deepmp_seq = [deepmp_seq_acc, precision, recall, f_score]
            import pdb;pdb.set_trace()
            if 'guppy_test' in locals():
                get_barplot_seq(
                    accs_deepmp, accs_deepmp_seq, accs_deepsignal, 
                    accs_nanopolish, accs_guppy, accs_megalodon, output
                )
                fig_out = os.path.join(output, 'comparison_deepmp_seq.pdf')
                plot_ROC_deepmpseq(
                    deepmp, deepmp_seq, merge, nanopolish_test, guppy_test, megalodon_test, fig_out
                )
                out_prere = os.path.join(output, 'AUC_prec_recall_deepmpseq.pdf')
                plot_precision_recall_curve_deepmpseq(deepmp, deepmp_seq, merge, nanopolish_test, guppy_test, megalodon_test, out_prere)
            else:
                
                get_barplot_seq_pUC19(
                    accs_deepmp, accs_deepmp_seq, accs_deepsignal, 
                    accs_nanopolish, accs_megalodon, output
                )
                fig_out = os.path.join(output, 'comparison_deepmp_seq_pUC19.pdf')
                plot_ROC_deepmpseq_pUC19(
                    deepmp, deepmp_seq, merge, nanopolish_test, megalodon_test, fig_out
                ) 
        
        if 'guppy_test' in locals():
            fig_out = os.path.join(output, 'comparison_megalodon.pdf')
            plot_ROC_megalodon(
                deepmp, merge, nanopolish_test, guppy_test, megalodon_test, fig_out
            )   
            
            out_prere = os.path.join(output, 'AUC_prec_recall_megalodon.pdf')
            plot_precision_recall_curve_megalodon(deepmp, merge, nanopolish_test, guppy_test, megalodon_test, out_prere)
        
        else:
            fig_out = os.path.join(output, 'comparison_megalodon.pdf')
            plot_ROC_megalodon_pUC19(
                deepmp, merge, nanopolish_test, megalodon_test, fig_out
            )
        
    
    # save_output([precision, recall, f_score], output) 
    if deepmod_output:
        fig_out = os.path.join(output, 'comparison_all.pdf')
        plot_ROC_all(deepmp, merge, deepmod, nanopolish_test, fig_out)   
        
        out_prere = os.path.join(output, 'AUC_prec_recall_all.pdf')
        plot_precision_recall_curve_all(deepmp, merge, deepmod, nanopolish_test, out_prere)

        if deepmod_accuracies:
            out_bar_all = os.path.join(output, 'accuracy_comparison_all.pdf')
            get_accuracies_all(deepmp_accuracies, deepmod_accuracies, merge, precision, recall, f_score, out_bar_all)




if __name__ == "__main__":
    main()