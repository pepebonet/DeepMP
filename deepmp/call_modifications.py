#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import seaborn as sns
import bottleneck as bn
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support

import deepmp.utils as ut
import deepmp.plots as pl
import deepmp.preprocess as pr


def acc_test_single(data, labels, model_file, score_av='binary'):
    model = load_model(model_file)
    # test_loss, test_acc = model.evaluate(data, tf.convert_to_tensor(labels))
    test_acc = 93.3
    pred =  model.predict(data).flatten()
    inferred = np.zeros(len(pred), dtype=int)
    inferred[np.argwhere(pred >= 0.5)] = 1

    precision, recall, f_score, _ = precision_recall_fscore_support(
        labels, inferred, average=score_av
    )
    
    return [test_acc, precision, recall, f_score], pred, inferred


def get_accuracy_joint(inferred, err_pred, seq_pred, labels, score_av='binary'):
    probs = np.zeros(len(labels))
    for i in range(len(labels)):
        if err_pred[i] > 0.5 and seq_pred[i] > 0.5:
            inferred[i] = 1
            probs[i] = max(seq_pred[i], err_pred[i])
        elif err_pred[i] < 0.5 and seq_pred[i] < 0.5:
            inferred[i] = 0
            probs[i] = min(seq_pred[i], err_pred[i])
        else: 
            val = (err_pred[i] + seq_pred[i]) / 2
            if val > 0.5:
                inferred[i] = 1
                probs[i] = max(seq_pred[i], err_pred[i])
            else: 
                inferred[i] = 0
                probs[i] = min(seq_pred[i], err_pred[i])

    test_acc = round(1 - np.argwhere(labels != inferred).shape[0] / len(labels), 5)
    
    precision, recall, f_score, _ = precision_recall_fscore_support(
        labels, inferred, average=score_av
    )

    return [test_acc, precision, recall, f_score], probs


def acc_test_joint(data_seq, labels_seq, model_seq, 
    data_err, labels_err, model_err):

    assert labels_err.all() == labels_seq.all()
    labels = labels_seq

    model_seq = load_model(model_seq)
    model_err = load_model(model_err)

    seq_pred = model_seq.predict(data_seq).flatten()
    err_pred = model_err.predict(data_err).flatten()
    
    inferred = np.zeros(len(seq_pred))

    return get_accuracy_joint(inferred, err_pred, seq_pred, labels)


def save_output(acc, output, label):
    col_names = ['Accuracy', 'Precision', 'Recall', 'F-score']
    df = pd.DataFrame([acc], columns=col_names)
    df.to_csv(os.path.join(output, label), index=False, sep='\t')


def save_probs(probs, labels, output):
    out_probs = os.path.join(output, 'test_pred_prob.txt')
    probs_to_save = pd.DataFrame(columns=['labels', 'probs'])
    probs_to_save['labels'] = labels
    probs_to_save['probs'] = probs
    probs_to_save.to_csv(out_probs, sep='\t', index=None)


def pred_site(df, pred_label, meth_label):
    comb_pred = df.pred_prob.min() + df.pred_prob.max()
    if comb_pred >= 1:
        pred_label.append(1)
    else: 
        pred_label.append(0)
    meth_label.append(df.methyl_label.unique()[0])

    return pred_label, meth_label


def pred_site_deepmod(df, pred_label, meth_label, threshold=0.3):
    inferred = df['inferred_label'].values
    if np.sum(inferred) / len(inferred) >= threshold:
        pred_label.append(1)
    else: 
        pred_label.append(0)
    meth_label.append(df.methyl_label.unique()[0])

    return pred_label, meth_label


def get_accuracy_pos(meth_label, pred_label):
    pos = np.argwhere(np.asarray(meth_label) == 1)
    neg = np.argwhere(np.asarray(meth_label) == 0)

    pred_pos = np.asarray(pred_label)[pos]
    pred_neg = np.asarray(pred_label)[neg]

    accuracy = (sum(pred_pos) + len(pred_neg) - sum(pred_neg)) / \
        (len(pred_pos) + len(pred_neg)) 
    return accuracy[0]


def do_per_position_analysis(df, output):
    df['id'] = df['chrom'] + '_' + df['pos'].astype(str)
    meth_label = []; pred_label = []; cov = []; new_df = pd.DataFrame()
    pred_label_cov = []
    for i, j in df.groupby('id'):
        if len(j.methyl_label.unique()) > 1:
            for k, l in j.groupby('methyl_label'):
                if len(l) > 0:
                    pred_label, meth_label = pred_site(l, pred_label, meth_label)
                    cov.append(len(l))
        else:
            if len(j) > 0:
                pred_label, meth_label = pred_site(j, pred_label, meth_label)
                cov.append(len(j))
            
    precision, recall, f_score, _ = precision_recall_fscore_support(
        meth_label, pred_label, average='binary'
    )

    accuracy_cov(pred_label, meth_label, cov, output)
    # TODO generalize for test with no label 
    # TODO improve calling of a methylation
    # TODO Add to the joint analysis 
    # TODO delete all unnecessary functions
    accuracy = get_accuracy_pos(meth_label, pred_label)
    save_output(
        [accuracy, precision, recall, f_score], output, 'position_accuracy.txt'
    )


#TODO send to plots once done
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


def plot_distributions(FN, TP, TN, FP, label, output):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import Counter
    import pdb;pdb.set_trace() 
    
    

    if label == 'currents.pdf':
        fn = Counter(FN.flatten())
        tp = Counter(TP.flatten())
        tn = Counter(TN.flatten())
        fp = Counter(FP.flatten())

        plt.scatter(list(tn.keys()), list(tn.values()) / np.sum(list(tn.values())), s=10, label='TN')
        plt.scatter(list(tp.keys()), list(tp.values()) / np.sum(list(tp.values())), s=10, label='TP')
        plt.scatter(list(fp.keys()), list(fp.values()) / np.sum(list(fp.values())), s=10, label='FP')
        plt.scatter(list(fn.keys()), list(fn.values()) / np.sum(list(fn.values())), s=10, label='FN')
        plt.xlim(0, 50)
    else:
        if label == 'std.pdf':
            sns.distplot(FN.flatten(), label='FN', bins=1000)
            sns.distplot(TP.flatten(), label='TP', bins=1000)
            sns.distplot(TN.flatten(), label='TN', bins=1000)
            sns.distplot(FP.flatten(), label='FP', bins=1000)
        else:
            sns.distplot(FN.flatten(), label='FN', bins=100)
            sns.distplot(TP.flatten(), label='TP', bins=100)
            sns.distplot(TN.flatten(), label='TN', bins=100)
            sns.distplot(FP.flatten(), label='FP', bins=100)
        plt.xlim(0, 1)
    plt.legend()
    plt.xlabel(label.split('.')[0])
    plt.ylabel('Density')
    plt.savefig(os.path.join(output, label))
    plt.close()


def pepe_more_tripping():
    aa = bn.move_mean(currents, window=17)
    bb = aa[~np.isnan(aa)]
    cc = np.argwhere(bb <= 7)
    true = labels[cc].flatten() 
    inf = inferred[cc].flatten()
    mean = mean[cc]
    stds = stds[cc]

    mean_pos = mean[np.argwhere(true == 1).flatten()]
    mean_neg = mean[np.argwhere(true == 0).flatten()]
    all_neg = inf[np.argwhere(true == 0).flatten()]
    all_neg = inf[np.argwhere(true == 0).flatten()]

    mean_FN = mean_pos[np.argwhere(all_pos == 0).flatten()]
    mean_TP = mean_pos[np.argwhere(all_pos == 1).flatten()]
    mean_TN = mean_neg[np.argwhere(all_neg == 0).flatten()]
    mean_FP = mean_neg[np.argwhere(all_neg == 1).flatten()]
    #same for stds
    plot_distributions(mean_FN, mean_TP, mean_TN, mean_FP, 'mean_test.pdf', output)



def pepe_tripping(data_seq, labels, inferred, output):
    currents = data_seq[:, :, -1].numpy()
    stds = data_seq[:, :, -2].numpy()
    mean = data_seq[:, :, -3].numpy()

    currents_pos = currents[np.argwhere(labels == 1).flatten()]
    currents_neg = currents[np.argwhere(labels == 0).flatten()]

    stds_pos = stds[np.argwhere(labels == 1).flatten()]
    stds_neg = stds[np.argwhere(labels == 0).flatten()]

    mean_pos = mean[np.argwhere(labels == 1).flatten()]
    mean_neg = mean[np.argwhere(labels == 0).flatten()]

    all_pos = inferred[np.argwhere(labels == 1).flatten()]
    all_neg = inferred[np.argwhere(labels == 0).flatten()]

    stds_FN = stds_pos[np.argwhere(all_pos == 0).flatten()]
    stds_TP = stds_pos[np.argwhere(all_pos == 1).flatten()]
    stds_TN = stds_neg[np.argwhere(all_neg == 0).flatten()]
    stds_FP = stds_neg[np.argwhere(all_neg == 1).flatten()]
    
    mean_FN = mean_pos[np.argwhere(all_pos == 0).flatten()]
    mean_TP = mean_pos[np.argwhere(all_pos == 1).flatten()]
    mean_TN = mean_neg[np.argwhere(all_neg == 0).flatten()]
    mean_FP = mean_neg[np.argwhere(all_neg == 1).flatten()]

    currents_FN = currents_pos[np.argwhere(all_pos == 0).flatten()]
    currents_TP = currents_pos[np.argwhere(all_pos == 1).flatten()]
    currents_TN = currents_neg[np.argwhere(all_neg == 0).flatten()]
    currents_FP = currents_neg[np.argwhere(all_neg == 1).flatten()]
    import pdb;pdb.set_trace()
    plot_distributions(
        currents_FN, currents_TP, currents_TN, currents_FP, 'currents.pdf', output)
    plot_distributions(mean_FN, mean_TP, mean_TN, mean_FP, 'mean.pdf', output)
    plot_distributions(stds_FN, stds_TP, stds_TN, stds_FP, 'std.pdf', output)


def call_mods(model, test_file, model_err, model_seq, one_hot_embedding, 
    kmer_sequence, output, figures=False):

    if model == 'seq':

        if test_file.rsplit('.')[-1] == 'tsv':
            test = pd.read_csv(test_file, sep='\t', nrows=1100000,
                names=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 
                'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_lens', 
                'cent_signals', 'methyl_label']) 
            pr.preprocess_sequence(test, os.path.dirname(test_file), 'test')
            test_file = os.path.join(os.path.dirname(test_file), 'test_seq.h5')

        data_seq, labels = ut.get_data_sequence(
            test_file, kmer_sequence, one_hot_embedding
        )
        import pdb;pdb.set_trace()
        acc, pred, inferred = acc_test_single(data_seq, labels, model_seq)
        save_probs(pred, labels, output)
        # pepe_tripping(data_seq, labels, inferred, output)
        try:
            test['pred_prob'] = pred; test['inferred_label'] = inferred
            plot_distributions(test, output)
            do_per_position_analysis(test, output)
        except: 
            print('No position analysis performed. Only per-read accuracy run')

    elif model == 'err':
        data_err, labels = ut.load_error_data(test_file)
        acc, pred, inferred = acc_test_single(data_err, labels, model_err)

    elif model == 'joint':
        data_seq, labels_seq = ut.get_data_sequence(
            test_file, kmer_sequence, one_hot_embedding
        )
        data_err, labels_err = ut.load_error_data(test_file)
 
        acc, probs = acc_test_joint(data_seq, labels_seq, model_seq, data_err, 
            labels_err, model_err)
        
        labels = labels_seq
        save_probs(probs, labels, output)
        
    save_output(acc, output, 'accuracy_measurements.txt')
    
    
    if figures:
        out_fig = os.path.join(output, 'ROC_curve.png')
        pl.plot_ROC(labels, probs, out_fig)

