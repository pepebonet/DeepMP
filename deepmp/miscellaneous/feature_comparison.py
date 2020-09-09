#!/usr/bin/envs python3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_ROC_seq(mean, median, std, skew, kurt, diff, currents, fig_out, kn='Linear'):

    fpr_mean, tpr_mean, thresholds = roc_curve(
        mean['labels'].values, mean['probs'].values
    )
    fpr_median, tpr_median, thresholds = roc_curve(
        median['labels'].values, median['probs'].values
    )
    fpr_std, tpr_std, thresholds = roc_curve(
        std['labels'].values, std['probs'].values
    )
    fpr_skew, tpr_skew, thresholds = roc_curve(
        skew['labels'].values, skew['probs'].values
    )
    fpr_kurt, tpr_kurt, thresholds = roc_curve(
        kurt['labels'].values, kurt['probs'].values
    )
    fpr_diff, tpr_diff, thresholds = roc_curve(
        diff['labels'].values, diff['probs'].values
    )
    fpr_currents, tpr_currents, thresholds = roc_curve(
        currents['labels'].values, currents['probs'].values
    )

    roc_auc_mean = auc(fpr_mean, tpr_mean)
    roc_auc_median = auc(fpr_median, tpr_median)
    roc_auc_std = auc(fpr_std, tpr_std)
    roc_auc_skew = auc(fpr_skew, tpr_skew)
    roc_auc_kurt = auc(fpr_kurt, tpr_kurt)
    roc_auc_diff = auc(fpr_diff, tpr_diff)
    roc_auc_currents = auc(fpr_currents, tpr_currents)

    plt.plot (fpr_mean, tpr_mean, label='mean: {}'.format(round(roc_auc_mean, 3)))
    plt.plot (fpr_median, tpr_median, label='median: {}'.format(round(roc_auc_median, 3)))
    plt.plot (fpr_std, tpr_std, label='std: {}'.format(round(roc_auc_std, 3)))
    plt.plot (fpr_skew, tpr_skew, label='skew: {}'.format(round(roc_auc_skew, 3)))
    plt.plot (fpr_kurt, tpr_kurt, label='kurt: {}'.format(round(roc_auc_kurt, 3)))
    plt.plot (fpr_diff, tpr_diff, label='diff: {}'.format(round(roc_auc_diff, 3)))
    plt.plot (fpr_currents, tpr_currents, label='currents: {}'.format(round(roc_auc_currents, 3)))


    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Ecoli data')
    plt.legend(loc="lower right")
    plt.savefig(fig_out)



def plot_ROC_err(all, quality, mis, ins, dele, fig_out, kn='Linear'):

    fpr_all, tpr_all, thresholds = roc_curve(
        all['labels'].values, all['probs'].values
    )
    fpr_quality, tpr_quality, thresholds = roc_curve(
        quality['labels'].values, quality['probs'].values
    )
    fpr_mis, tpr_mis, thresholds = roc_curve(
        mis['labels'].values, mis['probs'].values
    )
    fpr_ins, tpr_ins, thresholds = roc_curve(
        ins['labels'].values, ins['probs'].values
    )
    fpr_dele, tpr_dele, thresholds = roc_curve(
        dele['labels'].values, dele['probs'].values
    )

    roc_auc_all = auc(fpr_all, tpr_all)
    roc_auc_quality = auc(fpr_quality, tpr_quality)
    roc_auc_mis = auc(fpr_mis, tpr_mis)
    roc_auc_ins = auc(fpr_ins, tpr_ins)
    roc_auc_dele = auc(fpr_dele, tpr_dele)

    plt.plot (fpr_all, tpr_all, label='Combined: {}'.format(round(roc_auc_all, 3)))
    plt.plot (fpr_quality, tpr_quality, label='Quality: {}'.format(round(roc_auc_quality, 3)))
    plt.plot (fpr_mis, tpr_mis, label='Mismatches: {}'.format(round(roc_auc_mis, 3)))
    plt.plot (fpr_ins, tpr_ins, label='Insertions: {}'.format(round(roc_auc_ins, 3)))
    plt.plot (fpr_dele, tpr_dele, label='Deletions: {}'.format(round(roc_auc_dele, 3)))


    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Ecoli data')
    plt.legend(loc="lower right")
    plt.savefig(fig_out)


def main_sequence():
    mean = pd.read_csv('/workspace/projects/nanopore/DeepMP/outputs/test_features/only_mean/test_pred_prob.txt', sep='\t')
    median = pd.read_csv('/workspace/projects/nanopore/DeepMP/outputs/test_features/only_median/test_pred_prob.txt', sep='\t')
    std = pd.read_csv('/workspace/projects/nanopore/DeepMP/outputs/test_features/only_std/test_pred_prob.txt', sep='\t')
    skew = pd.read_csv('/workspace/projects/nanopore/DeepMP/outputs/test_features/only_skew/test_pred_prob.txt', sep='\t')
    kurt = pd.read_csv('/workspace/projects/nanopore/DeepMP/outputs/test_features/only_kurt/test_pred_prob.txt', sep='\t')
    diff = pd.read_csv('/workspace/projects/nanopore/DeepMP/outputs/test_features/only_diff/test_pred_prob.txt', sep='\t')
    currents = pd.read_csv('/workspace/projects/nanopore/DeepMP/outputs/test_features/only_currents/test_pred_prob.txt', sep='\t')

    fig_out = '/workspace/projects/nanopore/DeepMP/outputs/test_features/ROC_features.pdf'
    plot_ROC_seq(mean, median, std, skew, kurt, diff, currents, fig_out)


def main_errors():
    aall = pd.read_csv('/workspace/projects/nanopore/DeepMP/outputs/ecoli/error_features/all/test_pred_prob.txt', sep='\t')
    mis = pd.read_csv('/workspace/projects/nanopore/DeepMP/outputs/ecoli/error_features/only_mismatches/test_pred_prob.txt', sep='\t')
    ins = pd.read_csv('/workspace/projects/nanopore/DeepMP/outputs/ecoli/error_features/only_insertions/test_pred_prob.txt', sep='\t')
    dele = pd.read_csv('/workspace/projects/nanopore/DeepMP/outputs/ecoli/error_features/only_deletions/test_pred_prob.txt', sep='\t')
    qual = pd.read_csv('/workspace/projects/nanopore/DeepMP/outputs/ecoli/error_features/only_quality/test_pred_prob.txt', sep='\t')

    fig_out = '/workspace/projects/nanopore/DeepMP/outputs/ecoli/error_features/ROC_features.pdf'
    plot_ROC_err(aall, qual, mis, ins, dele, fig_out)


if __name__ == '__main__':
    # main_sequence()

    main_errors()