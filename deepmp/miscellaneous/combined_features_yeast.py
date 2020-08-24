#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from functools import reduce
from tensorflow.keras.models import load_model

def get_id(df, name):
    df['id'] = df['Ref'] + '_' + df['pos'].astype(str)
    df['labels'] = 0; df['name'] = name
    return df[df['Cov'] >= 1]


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
    return pred


def predict_seq(df, model):
    pass


def get_mod_score(s1, s2, s3):
    Mod_score = []
    for i in range(len(s1)):
        if s1[i] >= 0.5 and s2[i] >= 0.5 and s3[i] >= 0.5:
            Mod_score.append(1)
        else:
            Mod_score.append((s1[i] + s2[i] + s3[i]) / 3)
    return Mod_score


def call_mod(ko, wt):
    mods = []
    for i in range(len(ko)):
        if ko[i] / wt[i] > 1.5 and ko[i] > 0.5:
            mods.append(1)
        else:
            mods.append(0)
    import pdb; pdb.set_trace()
    return mods


def main(ko1, ko2, ko3, wt1, wt2, wt3, error_model_file, seq_model_file, wt1_seq):

    ko1, ko2, ko3, wt1, wt2, wt3 = merge_all(ko1, ko2, ko3, wt1, wt2, wt3)
    import pdb;pdb.set_trace()
    error_model = load_model(error_model_file)
    err_pred_ko1 = predict_error(ko1, error_model)    
    err_pred_ko2 = predict_error(ko2, error_model)
    err_pred_ko3 = predict_error(ko3, error_model)
    err_pred_wt1 = predict_error(wt1, error_model)
    err_pred_wt2 = predict_error(wt2, error_model)
    err_pred_wt3 = predict_error(wt3, error_model)

    seq_model = load_model(seq_model_file)
    seq_pred_ko1 = predict_seq(ko1, seq_model)    
    seq_pred_ko2 = predict_seq(ko2, seq_model)
    seq_pred_ko3 = predict_seq(ko3, seq_model)
    seq_pred_wt1 = predict_seq(wt1, seq_model)
    seq_pred_wt2 = predict_seq(wt2, seq_model)
    seq_pred_wt3 = predict_seq(wt3, seq_model)

    ko_mod_score = get_mod_score(err_pred_ko1, err_pred_ko2, err_pred_ko3)
    wt_mod_score = get_mod_score(err_pred_wt1, err_pred_wt2, err_pred_wt3)
    
    final_mod_score = call_mod(ko_mod_score, wt_mod_score)
    import pdb;pdb.set_trace()
    

if __name__ == "__main__":
    err_model_file = '/workspace/projects/nanopore/DeepMP/models/epinano_combined/error_model/'  
    bd = '/workspace/projects/nanopore/stockholm/EpiNano/novoa_features/yeast_epinano/'
    ko1_err = get_id(pd.read_csv(os.path.join(bd, 'KO1/RRACH_5x_KO1.tsv'), sep='\t'), 'ko1')
    ko2_err = get_id(pd.read_csv(os.path.join(bd, 'KO2/RRACH_5x_KO2.tsv'), sep='\t'), 'ko2')
    ko3_err = get_id(pd.read_csv(os.path.join(bd, 'KO3/RRACH_5x_KO3.tsv'), sep='\t'), 'ko3')
    wt1_err = get_id(pd.read_csv(os.path.join(bd, 'wildtype1/RRACH_5x_WT1.tsv'), sep='\t'), 'wt1')
    wt2_err = get_id(pd.read_csv(os.path.join(bd, 'wildtype2/RRACH_5x_WT2.tsv'), sep='\t'), 'wt2')
    wt3_err = get_id(pd.read_csv(os.path.join(bd, 'wildtype3/RRACH_5x_WT3.tsv'), sep='\t'), 'wt3')

    seq_model_file = '/workspace/projects/nanopore/DeepMP/models/epinano_combined/sequence_model/'
    # ko1_seq = pd.read_csv(os.path.join(bd, 'KO1/sequence_features/feature_KO1s_RRACH.tsv'), sep='\t')
    # ko2_seq = pd.read_csv(os.path.join(bd, 'KO2/sequence_features/feature_KO2s_RRACH.tsv'), sep='\t')
    # ko3_seq = pd.read_csv(os.path.join(bd, 'KO3/sequence_features/feature_KO3s_RRACH.tsv'), sep='\t')
    wt1_seq = pd.read_csv(os.path.join(bd, 'wildtype1/sequence_features/features_WT1_RRACH.tsv'), sep='\t', header=None)
    # wt2_seq = pd.read_csv(os.path.join(bd, 'wildtype2/sequence_features/feature_WT2s_RRACH.tsv'), sep='\t')
    # wt3_seq = pd.read_csv(os.path.join(bd, 'wildtype3/sequence_features/feature_WT3s_RRACH.tsv'), sep='\t')

    main(ko1_err, ko2_err, ko3_err, wt1_err, wt2_err, wt3_err, err_model_file, seq_model_file, wt1_seq)