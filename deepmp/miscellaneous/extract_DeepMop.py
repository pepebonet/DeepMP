#!/usr/bin/env python3
import os
import h5py
import click
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support


def get_label(treatment):
    if treatment == 'untreated':
        return 0
    else: 
        return 1 


def get_reads_info(fast5, index, label):
    ind = pd.read_csv(index, sep=' ', header=None).values[:, [3,4]]

    true = [] ; pred = []
    with h5py.File(fast5, 'r') as hf:
        for el in ind: 
            data = hf['pred/{}/predetail'.format(el[0])][:]

            refseq = ''.join([x[0].astype(str) for x in data])
            generator = slice_chunks(refseq, 2)
            chunks = np.asarray(list(generator))

            df = pd.DataFrame(data[np.argwhere(chunks == 'CG')].flatten())

            if label == 1:
                pred.append(df['mod_pred'].values)
                true.append(np.ones(len(df['mod_pred'].values)))
                import pdb;pdb.set_trace()
            else:
                pred.append(df['mod_pred'].values)
                true.append(np.zeros(len(df['mod_pred'].values)))
    return true, pred


def slice_chunks(l, n):
    for i in range(0, len(l) - n + 1):
        yield l[i:i + n]



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
    '-o', '--output', default='', help='output folder'
)
def main(detect_folder, file_id, label, output):
    
    treat_true = []; treat_pred = []
    untreat_true = []; untreat_pred = []
    treatments = os.listdir(detect_folder)
    for treat in treatments:
        label = get_label(treat)
        detect_subfolder = os.path.join(detect_folder, treat, file_id)
        detect_reads = os.listdir(detect_subfolder)
        for el in detect_reads: 
            if os.path.isdir(os.path.join(detect_subfolder, el)):
                read_folder = os.path.join(detect_subfolder, el)
                reads = os.listdir(read_folder)

                for i in range(int(len(reads)/2)):
                    fast5 = os.path.join(
                        read_folder, 'rnn.pred.detail.fast5.{}'.format(i)
                    )
                    index = os.path.join(
                        read_folder, 'Chromosome.rnn.pred.ind.{}'.format(i)
                    )
                    true, pred = get_reads_info(fast5, index, label)
                    # import pdb;pdb.set_trace()
                    if label == 1:
                        treat_true.append(np.concatenate(true))
                        treat_pred.append(np.concatenate(pred))
                    else:
                        untreat_true.append(np.concatenate(true))
                        untreat_pred.append(np.concatenate(pred))

    treat_true = np.concatenate(treat_true)
    treat_pred = np.concatenate(treat_pred)
    untreat_true = np.concatenate(untreat_true)
    untreat_pred = np.concatenate(untreat_pred)

    true = np.concatenate([treat_true, untreat_true])
    pred = np.concatenate([treat_pred, untreat_pred])

    precision, recall, f_score, _ = precision_recall_fscore_support(
        true, pred, average='binary'
    )
    print(precision, recall, f_score)
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()