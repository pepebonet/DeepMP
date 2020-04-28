#!/usr/bin/env python3
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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