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

    # treat_pred = []; treat_true = []
    # untreat_pred = []; untreat_true = []
    true = [] ; pred = []
    with h5py.File(fast5, 'r') as hf:
        for el in ind: 
            data = hf['pred/{}/predetail'.format(el[0])][:]

            refseq = ''.join([x[0].astype(str) for x in data])
            generator = slice_chunks(refseq, 2)
            chunks = np.asarray(list(generator))

            df = pd.DataFrame(data[np.argwhere(chunks == 'CG')].flatten())

            if label == 1:
                print('Modified')
                print(df['mod_pred'].sum() / df.shape[0])
                pred.append(df['mod_pred'].values)
                true.append(np.ones(len(df['mod_pred'].values)))
            else:
                print('Unmodified')
                print((df.shape[0] - df['mod_pred'].sum()) / df.shape[0])
                pred.append(df['mod_pred'].values)
                true.append(np.zeros(len(df['mod_pred'].values)))
    

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
    
    treatments = os.listdir(detect_folder)
    for treat in treatments:
        label = get_label(treat)
        detect_subfolder = os.path.join(detect_folder, treat, file_id)
        detect_reads = os.listdir(detect_subfolder)
        for el in detect_reads: 
            if os.path.isdir(os.path.join(detect_subfolder, el)):
                read_folder = os.path.join(detect_subfolder, el)
                reads = os.listdir(read_folder)
                for i in range(1000):
                    matching = [s for s in os.listdir(read_folder) if str(i) in s]
                    if matching: 
                        fast5 = os.path.join(read_folder, matching[0])
                        index = os.path.join(read_folder, matching[1])
                        # true, pred = 
                        import pdb;pdb.set_trace()
    
    import pdb;pdb.set_trace()

    treat_true = np.concatenate(treat_true)
    treat_pred = np.concatenate(treat_pred)
    untreat_true = np.concatenate(untreat_true)
    untreat_pred = np.concatenate(untreat_pred)
    import pdb;pdb.set_trace()
    true = np.concatenate([treat_true, untreat_true])
    pred = np.concatenate([treat_pred, untreat_pred])

    precision, recall, f_score, _ = precision_recall_fscore_support(
        true, pred, average='binary'
    )
    print(precision, recall, f_score)
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()