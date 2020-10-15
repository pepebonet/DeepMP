#!/usr/bin/env python3
import os
import sys
import h5py
import click
import functools
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.metrics import precision_recall_fscore_support

sys.path.append('../')
import deepmp.utils as ut



def get_test_df(test_file):
    test = pd.read_csv(test_file, sep='\t',
        names=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 
        'read_strand', 'kmer', 'methyl_label'])
    #Pos in strand = pos in ref. Should be the correct choice for position

    test['id'] = test['pos'].astype(str) \
        + '_' + test['chrom'] + '_' + test['strand']

    return test


def pred_site(df, pred_label, meth_label, threshold=0.2):
    inferred = df['mod_pred'].values
    if np.sum(inferred) / len(inferred) >= threshold:
        pred_label.append(1)
    else: 
        pred_label.append(0)
    meth_label.append(df.methyl_label.unique()[0])

    return pred_label, meth_label


def do_per_position_analysis(df, output):
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
    accuracy = get_accuracy_pos(meth_label, pred_label)
    save_output(
        [accuracy, precision, recall, f_score], output, 'position_accuracy.txt'
    )


def get_accuracy_pos(meth_label, pred_label):
    pos = np.argwhere(np.asarray(meth_label) == 1)
    neg = np.argwhere(np.asarray(meth_label) == 0)

    pred_pos = np.asarray(pred_label)[pos]
    pred_neg = np.asarray(pred_label)[neg]

    accuracy = (sum(pred_pos) + len(pred_neg) - sum(pred_neg)) / \
        (len(pred_pos) + len(pred_neg)) 
    return accuracy[0]


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
    # ax.set_ylim(0.9,1)
    fig.tight_layout()
    
    plt.savefig(os.path.join(output, 'acc_vs_cov.png'))
    plt.close()


def get_plots_and_accuracies(treat_true, treat_pred, untreat_true, untreat_pred,
    output, test_all):
    treat_true = np.concatenate(treat_true)
    treat_pred = np.concatenate(treat_pred)
    untreat_true = np.concatenate(untreat_true)
    untreat_pred = np.concatenate(untreat_pred)

    true = np.concatenate([treat_true, untreat_true])
    pred = np.concatenate([treat_pred, untreat_pred])

    precision, recall, f_score, _ = precision_recall_fscore_support(
        true, pred, average='binary'
    )
    accuracy = get_accuracy_pos(true, pred)
    save_output(
        [accuracy, precision, recall, f_score], output, 'accuracy_all.txt'
    )
    print(precision, recall, f_score)
    
    ut.save_probs(pred, true, output)

    if not test_all.empty:
        do_per_position_analysis(test_all, output)


def save_output(acc, output, label):
    col_names = ['Accuracy', 'Precision', 'Recall', 'F-score']
    df = pd.DataFrame([acc], columns=col_names)
    df.to_csv(os.path.join(output, label), index=False, sep='\t')


def get_label(treatment):
    if treatment == 'untreated':
        return 0
    else: 
        return 1 


def get_reads_info(index, test, base_folder):
    fast5 = os.path.join(base_folder, index.values[0][5])
    true = [] ; pred = []; df_test = pd.DataFrame()

    with h5py.File(fast5, 'r') as hf:

        data = hf['pred/{}/predetail'.format(index.values[0][3])][:]
        df = pd.DataFrame(data)
        df = df[df['refbase'] != b'-']

        df['id'] = df['refbasei'].astype(str) + '_' + \
            index.values[0][0] + '_' + index.values[0][1] 
        merged = pd.merge(df, test, how='inner', on='id')

        pred = merged['mod_pred'].values
        true = merged['methyl_label'].values

        return true, pred, merged


def do_read_analysis(el, test, ind, detect_subfolder):
    sub_test = test[test['readname'] == el]
    sub_ind = ind[ind[4] == el]

    if sub_ind.shape[0] > 0:
        return get_reads_info(sub_ind, sub_test, detect_subfolder)
    else:
        print('No index available! ')


@click.command(short_help='Convert DeepMod output into accuracy scores.')
@click.option(
    '-df', '--detect_folder', required=True, 
    help='Folder containing the test set'
)
@click.option(
    '-fid', '--file_id', required=True, 
    help='File ID for tombo detect. Default= User_Uniq_name'
)
@click.option(
    '-tf', '--test_file', required=True, 
    help='Test file to save only given positions.'
)
@click.option(
    '-cpu', '--cpus', default=1, 
    help='Select number of cpus to run'
)
@click.option(
    '-o', '--output', default='', help='output folder'
)
def main(detect_folder, file_id, output, test_file, cpus):

    test = get_test_df(test_file)
    
    treat_true = []; treat_pred = []
    untreat_true = []; untreat_pred = []
    test_all = pd.DataFrame()
    treatments = os.listdir(detect_folder)

    
    for treat in treatments:

        label = get_label(treat)
        detect_subfolder = os.path.join(detect_folder, treat, file_id)
        parser_file = os.path.join(detect_subfolder, 'rnn.pred.ind.Chromosome')
        ind = pd.read_csv(parser_file, sep=' ', header=None, skiprows=2)

        test_label = test[test['methyl_label'] == label]
        readnames = list(set(test_label.readname.to_list()))

        for el in tqdm(readnames):
            rval = do_read_analysis(el, test, ind, detect_subfolder)

            if rval is not None:   
                test_all = pd.concat([test_all, rval[2]])
                if label == 1:
                    treat_true.append(rval[0])
                    treat_pred.append(rval[1])
                else:
                    untreat_true.append(rval[0])
                    untreat_pred.append(rval[1])
    
    get_plots_and_accuracies(
        treat_true, treat_pred, untreat_true, untreat_pred, output, test_all
    )


if __name__ == '__main__':
    main()