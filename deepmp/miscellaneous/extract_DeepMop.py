#!/usr/bin/env python3
import os
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


def get_test_df(test_file):
    test = pd.read_csv(test_file, sep='\t',
        names=['chrom', 'pos', 'strand', 'pos_in_strand', 'readname', 
        'read_strand', 'kmer', 'methyl_label'])
    #Pos in strand = pos in ref. Should be the correct choice for position
    test['id'] = test['readname'] + '_' + test['pos_in_strand'].astype(str) \
        + '_' + test['chrom'] + '_' + test['strand']

    return test


def get_label(treatment):
    if treatment == 'untreated':
        return 0
    else: 
        return 1 


def get_list(reads, read_folder):
    fast5_index = []
    for i in range(int(len(reads)/2)):
        fast5 = os.path.join(
            read_folder, 'rnn.pred.detail.fast5.{}'.format(i)
        )
        index = os.path.join(
            read_folder, 'Chromosome.rnn.pred.ind.{}'.format(i)
        )
        fast5_index.append((fast5, index))

    return fast5_index


def do_read_analysis(reads, label, test):
    fast5 = reads[0]; index = reads[1]
    return get_reads_info(fast5, index, label, test)


def get_reads_info(fast5, index, label, test=None):
    ind = pd.read_csv(index, sep=' ', header=None)

    if test is not None: 
        merged_reads = pd.merge(ind, test, right_on='readname', left_on=4, how='inner')
        if merged_reads.shape[0] == 0:
            return [], [], pd.DataFrame()

    true = [] ; pred = []; df_test = pd.DataFrame()
    with h5py.File(fast5, 'r') as hf:
        for el in merged_reads.values: 

            data = hf['pred/{}/predetail'.format(el[3])][:]

            refseq = ''.join([x[0].astype(str) for x in data])
            generator = slice_chunks(refseq, 2)
            chunks = np.asarray(list(generator))

            df = pd.DataFrame(data[np.argwhere(chunks == 'CG')].flatten())
            df['readname'] = el[4] 
            df['chrom'] = el[0] 
            df['strand'] = el[1]       
            df['id'] = df['readname'] + '_' + df['refbasei'].astype(str) + '_' + \
                df['chrom'] + '_' + df['strand']

            if test is not None: 
                test_label = test[test['methyl_label'] == label]
                merged = pd.merge(df, test_label, how='inner', on='id')

                if not merged.empty:
                    # import pdb;pdb.set_trace()
                    pred.append(merged['mod_pred'].values)
                    true.append(merged['methyl_label'].values)
                    df_test = pd.concat([df_test, merged])

            else:
                if label == 1:
                    pred.append(df['mod_pred'].values)
                    true.append(np.ones(len(df['mod_pred'].values)))
                else:
                    pred.append(df['mod_pred'].values)
                    true.append(np.zeros(len(df['mod_pred'].values)))
    
    return true, pred, df_test

    


def slice_chunks(l, n):
    for i in range(0, len(l) - n + 1):
        yield l[i:i + n]


def pred_site(df, pred_label, meth_label, threshold=0.2):
    inferred = df['mod_pred'].values
    if np.sum(inferred) / len(inferred) >= threshold:
        pred_label.append(1)
    else: 
        pred_label.append(0)
    meth_label.append(df.methyl_label.unique()[0])

    return pred_label, meth_label


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

    if not test_all.empty:
        do_per_position_analysis(test_all, output)


def save_output(acc, output, label):
    col_names = ['Accuracy', 'Precision', 'Recall', 'F-score']
    df = pd.DataFrame([acc], columns=col_names)
    df.to_csv(os.path.join(output, label), index=False, sep='\t')


@click.command(short_help='Convert DeepMod output into accuracy scores.')
@click.option(
    '-df', '--detect_folder', required=True, 
    help='Folder containing the test set'
)
@click.option(
    '-fid', '--file_id', required=True, 
    help='File ID for tombo detect. Default= User_unique_name'
)
@click.option(
    '-l', '--label', default='', 
    help='Index file containing read names.'
)
@click.option(
    '-tf', '--test_file', default='', 
    help='Test file to save only given positions.'
)
@click.option(
    '-cpu', '--cpus', default=1, 
    help='Select number of cpus to run'
)
@click.option(
    '-o', '--output', default='', help='output folder'
)
def main(detect_folder, file_id, label, output, test_file, cpus):
    #TODO adapt for human
    test = None
    if test_file:
        test = get_test_df(test_file)
    
    treat_true = []; treat_pred = []
    untreat_true = []; untreat_pred = []
    test_all = pd.DataFrame()
    treatments = os.listdir(detect_folder)

    for treat in treatments:

        label = get_label(treat)
        detect_subfolder = os.path.join(detect_folder, treat, file_id)
        detect_reads = os.listdir(detect_subfolder)

        for el in detect_reads: 
            if os.path.isdir(os.path.join(detect_subfolder, el)):

                read_folder = os.path.join(detect_subfolder, el)
                reads = os.listdir(read_folder)
                fast5_index = get_list(reads, read_folder)

                f = functools.partial(do_read_analysis, label=label, test=test)

                with Pool(cpus) as p:
                    for i, rval in enumerate(p.imap_unordered(f, fast5_index)):
                        print('Completed: ' +  str(round(i/ len(fast5_index), 3)) + ' %')
                        
                        if rval[2].shape[0] == 0:
                            continue

                        test_all = pd.concat([test_all, rval[2]])
                        if label == 1:
                            treat_true.append(np.concatenate(rval[0]))
                            treat_pred.append(np.concatenate(rval[1]))
                        else:
                            untreat_true.append(np.concatenate(rval[0]))
                            untreat_pred.append(np.concatenate(rval[1]))

    get_plots_and_accuracies(
        treat_true, treat_pred, untreat_true, untreat_pred,output, test_all
    )


if __name__ == '__main__':
    main()