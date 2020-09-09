#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from functools import reduce
from collections import Counter
from tensorflow.keras.models import load_model

import sys
sys.path.append('../.')
import deepmp.preprocess as pr
import deepmp.utils as ut

def get_id(df, name):
    df['id'] = df['Ref'] + '_' + df['pos'].astype(str)
    df['labels'] = 0; df['name'] = name
    return df[df['Cov'] >= 1]


def filter_and_id(df, pos):
    df = df[df['strand'] == '+']
    df['id'] = df['chrom'] + '_' + (df['pos'] + 1).astype(str)
    return pd.merge(df, pos, on='id', how='inner').drop(columns=['id'])


def merge_all(ko1, ko2, ko3, wt1, wt2, wt3):
    dfs = [ko1, ko2, ko3, wt1, wt2, wt3]

    f_merged = reduce(
        lambda  left, right: pd.merge(left,right,on=['id'], how='inner'), dfs
    )
    print(f_merged.shape)

    ko1 = f_merged.iloc[:, 0:27]
    ko2 = f_merged.iloc[:, 27:53]
    ko3 = f_merged.iloc[:, 53:79]
    wt1 = f_merged.iloc[:, 79:105]
    wt2 = f_merged.iloc[:, 105:131]
    wt3 = f_merged.iloc[:, 131:158]
    return ko1, ko2, ko3, wt1, wt2, wt3

#TODO you get the merged table and separate them one more time (first). Then you predict for every table and do as in epinano
#that way you might be able to now how many positives and negatives you have (only with the erros)
#you should be able to combine it with the sequences and also use call modifications for that: 
# need to prepare the features in the h5 files (2) and then improve/change the call modifications file 
def get_X_Y(df):
    return df[df.columns[2:22]], df[df.columns[-1]]


def predict_error(df, model):
    try: 
        X = df[df.columns[2:22]].drop(columns=['ins1_x', 'ins2_x', 'ins3_x', 'ins4_x', 'ins5_x'])
    except:
        X = df[df.columns[2:22]].drop(columns=['ins1_y', 'ins2_y', 'ins3_y', 'ins4_y', 'ins5_y'])
    X = X.to_numpy().reshape(X.shape[0], 15, 1)
    pred =  model.predict(X).flatten()
    inferred = np.zeros(len(pred), dtype=int)
    inferred[np.argwhere(pred >= 0.5)] = 1
    print(sum(inferred))
    return pred, inferred


def predict_seq(test_file, model, pos):

    test = filter_and_id(pd.read_csv(test_file, sep='\t', nrows=10000,
            names=['chrom', 'pos', 'strand', 'pos_in_ref', 'readname', 
            'read_strand', 'kmer', 'signal_means', 'signal_stds', 'signal_lens', 
            'cent_signals', 'methyl_label']), pos)

    pr.preprocess_sequence(test, os.path.dirname(test_file), 'test')
    test_file = os.path.join(os.path.dirname(test_file), 'test_seq.h5')

    data_seq, labels = ut.get_data_sequence(
        test_file, 17, True
    )
    pred =  model.predict(data_seq).flatten()
    import pdb;pdb.set_trace()
    inferred = np.zeros(len(pred), dtype=int)
    inferred[np.argwhere(pred >= 0.5)] = 1
    test['pred_prob'] = pred; test['inferred_label'] = inferred
    print(sum(inferred))
    
    plot_distributions(test, output)
    do_per_position_analysis(test, output)


def get_mod_score(s1, s2, s3):
    Mod_score = []
    for i in range(len(s1)):
        if s1[i] >= 0.5 and s2[i] >= 0.5 and s3[i] >= 0.5:
            Mod_score.append(1)
        # elif s1[i] <= 0.5 and s2[i] <= 0.5 and s3[i] <= 0.5:
        #     Mod_score.append(0)
        else:
            Mod_score.append((s1[i] + s2[i] + s3[i]) / 3)

    return Mod_score

# def call_mod(ko, wt):
#     mods = []
#     for i in range(len(ko)):
#         if wt[i] == 0:
#             mods.append(0)
#         else:
#             if ko[i] == 0:
#                 if wt[i] > 0.5:
#                     mods.append(1)
#                 else:
#                     mods.append(0)
#             else:
#                 if wt[i] / ko[i] > 1 and wt[i] > 0.5:
#                     mods.append(1)
#                 else:
#                     mods.append(0)
#     return mods

def call_mod(ko, wt):
    mods = []
    for i in range(len(ko)):
        if wt[i] / ko[i] > 1 and wt[i] > 0.5:
            mods.append(1)
        else:
            mods.append(0)
    return mods


def main(ko1, ko2, ko3, wt1, wt2, wt3, error_model_file, pos, seq_model_file, 
    ko1_seq, ko2_seq, ko3_seq, wt1_seq, wt2_seq, wt3_seq, genes):

    error_model = load_model(error_model_file)
    err_pred_ko1, inf_ko1 = predict_error(ko1, error_model)    
    err_pred_ko2, inf_ko2 = predict_error(ko2, error_model)
    err_pred_ko3, inf_ko3 = predict_error(ko3, error_model)
    err_pred_wt1, inf_wt1 = predict_error(wt1, error_model)
    err_pred_wt2, inf_wt2 = predict_error(wt2, error_model)
    err_pred_wt3, inf_wt3 = predict_error(wt3, error_model)

    seq_model = load_model(seq_model_file)
    # seq_pred_ko1 = predict_seq(ko1_seq, seq_model, pos)    
    # seq_pred_ko2 = predict_seq(ko2_seq, seq_model, pos)
    # seq_pred_ko3 = predict_seq(ko3_seq, seq_model, pos)
    # seq_pred_wt1 = predict_seq(wt1_seq, seq_model, pos)
    # seq_pred_wt2 = predict_seq(wt2_seq, seq_model, pos)
    # seq_pred_wt3 = predict_seq(wt3_seq, seq_model, pos)

    ko_mod_score = get_mod_score(err_pred_ko1, err_pred_ko2, err_pred_ko3)
    wt_mod_score = get_mod_score(err_pred_wt1, err_pred_wt2, err_pred_wt3)
    
    final_mod_score = call_mod(ko_mod_score, wt_mod_score)

    ko1['mod'] = final_mod_score
    test = pd.merge(ko1, genes, on='id', how='inner')
    print(test['mod'].sum()) 
    import pdb;pdb.set_trace()


def do_analysis(df, genes, error_model):
    X = df[df.columns[2:22]].drop(columns=['ins1', 'ins2', 'ins3', 'ins4', 'ins5'])
    X = X.to_numpy().reshape(X.shape[0], 15, 1)
    model = load_model(error_model)
    pred =  model.predict(X).flatten()
    inferred = np.zeros(len(pred), dtype=int)
    inferred[np.argwhere(pred >= 0.5)] = 1
    df['mod'] = inferred ; df['pred'] = pred
    test = pd.merge(df, genes, on='id', how='inner')
    print(sum(inferred), test['mod'].sum(), test.shape)

    return df, test


def get_results(df):
    mod_list = [df['pred'], df['pred_x'], df['pred_y']]
    av = np.average(np.asarray(list(filter(lambda x: x != -1, mod_list))))
    if av > 0.5:
        return 1
    else: 
        return 0


def get_results_both(df):
    # import pdb;pdb.set_trace()
    if df['result_x'] == 1 and df['result_y'] == 0:
        return 1
    else:
        return 0


def do_genes_subset(df):
    if Counter(df['fullcons'][2:7])['A'] > 1:
        return 0
    else:
        return 1



def do_ind_analyses(ko1_err, ko2_err, ko3_err, wt1_err, wt2_err, wt3_err, genes, error_model):
    ko1, ko1_genes = do_analysis(ko1_err, genes, error_model)
    ko2, ko2_genes = do_analysis(ko2_err, genes, error_model)
    ko3, ko3_genes = do_analysis(ko3_err, genes, error_model)
    wt1, wt1_genes = do_analysis(wt1_err, genes, error_model)
    wt2, wt2_genes = do_analysis(wt2_err, genes, error_model)
    wt3, wt3_genes = do_analysis(wt3_err, genes, error_model)
    test_wt = pd.merge(wt1, pd.merge(wt2, wt3, on='id', how='outer'), on='id', how='outer').fillna(-1)
    test_ko = pd.merge(ko1, pd.merge(ko2, ko3, on='id', how='outer'), on='id', how='outer').fillna(-1)

    test_wt['result'] = test_wt.apply(get_results, axis=1)
    test_ko['result'] = test_ko.apply(get_results, axis=1)
    genes_wt = pd.merge(genes, test_wt, on='id', how='inner')
    genes_ko = pd.merge(genes, test_ko, on='id', how='inner')
    print(genes_wt.result.sum()); print(test_wt.shape)
    print(genes_ko.result.sum()); print(test_ko.shape)

    test_both = pd.merge(test_wt, test_ko, on='id', how='inner')
    test_both['result'] = test_both.apply(get_results_both, axis=1)
    aa = pd.merge(test_both, genes, on='id', how='inner') 
    print(aa.shape)
    #TODO <Start from scratch. Extract features. Train only on those that has one A. compare and then test.
    import pdb;pdb.set_trace()


    

if __name__ == "__main__":
    err_model_file = '/workspace/projects/nanopore/DeepMP/models/error_model/'  
    bd = '/workspace/projects/nanopore/stockholm/EpiNano/novoa_features/yeast_epinano/'
    ko1_err = get_id(pd.read_csv(os.path.join(bd, 'KO1/RRACH_5x_errors.tsv'), sep='\t'), 'ko1')
    ko2_err = get_id(pd.read_csv(os.path.join(bd, 'KO2/RRACH_5x_errors.tsv'), sep='\t'), 'ko2')
    ko3_err = get_id(pd.read_csv(os.path.join(bd, 'KO3/RRACH_5x_errors.tsv'), sep='\t'), 'ko3')
    wt1_err = get_id(pd.read_csv(os.path.join(bd, 'wildtype1/RRACH_5x_errors.tsv'), sep='\t'), 'wt1')
    wt2_err = get_id(pd.read_csv(os.path.join(bd, 'wildtype2/RRACH_5x_errors.tsv'), sep='\t'), 'wt2')
    wt3_err = get_id(pd.read_csv(os.path.join(bd, 'wildtype3/RRACH_5x_errors.tsv'), sep='\t'), 'wt3')

    genes = pd.read_csv(os.path.join(bd, 'sk1_highconf_m6Asites.txt'), sep='\t')
    genes['id'] = genes['chr'] + '_' + genes['peakgenomepos'].astype(str)
    genes['select'] = genes.apply(do_genes_subset, axis=1)
    genes = genes[genes['select'] == 1].drop(columns=['select']) 
    import pdb;pdb.set_trace()
    # do_ind_analyses(ko1_err, ko2_err, ko3_err, wt1_err, wt2_err, wt3_err, genes, err_model_file)
    

    ko1_err, ko2_err, ko3_err, wt1_err, wt2_err, wt3_err = merge_all(
        ko1_err, ko2_err, ko3_err, wt1_err, wt2_err, wt3_err)
    pos = ko1_err['id']
    
    
    seq_model_file = '/workspace/projects/nanopore/DeepMP/models/epinano_combined/sequence_model/'
    ko1_seq = os.path.join(bd, 'internet_genome/KO1/sequence_features/features_KO1_RRACH.tsv')
    ko2_seq = os.path.join(bd, 'internet_genome/KO2/sequence_features/features_KO2_RRACH.tsv')
    ko3_seq = os.path.join(bd, 'internet_genome/KO3/sequence_features/features_KO3_RRACH.tsv')
    wt1_seq = os.path.join(bd, 'internet_genome/wildtype1/sequence_features/features_WT1_RRACH.tsv')
    wt2_seq = os.path.join(bd, 'internet_genome/wildtype2/sequence_features/feature_WT2_RRACH.tsv')
    wt3_seq = os.path.join(bd, 'internet_genome/wildtype3/sequence_features/feature_WT3_RRACH.tsv')

    main(ko1_err, ko2_err, ko3_err, wt1_err, wt2_err, wt3_err, err_model_file, pos,\
        seq_model_file, ko1_seq, ko2_seq, ko3_seq, wt1_seq, wt2_seq, wt3_seq, genes)
