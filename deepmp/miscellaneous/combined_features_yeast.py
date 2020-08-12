#!/usr/bin/env python3

import os
import pandas as pd
from functools import reduce
from tensorflow.keras.models import load_model

def get_id(df):
    df['id'] = df['Ref'] + '_' + df['pos'].astype(str)
    df['labels'] = 0
    return df[df['Cov'] >= 1]


def merge_all(ko1, ko2, ko3, wt1, wt2, wt3):
    dfs = [ko1, ko2, ko3, wt1, wt2, wt3]

    f_merged = reduce(
        lambda  left, right: pd.merge(left,right,on=['id'], how='inner'), dfs
    )
    print(f_merged.shape)
    return f_merged

#TODO you get the merged table and separate them one more time (first). Then you predict for every table and do as in epinano
#that way you might be able to now how many positives and negatives you have (only with the erros)
#you should be able to combine it with the sequences and also use call modifications for that: 
# need to prepare the features in the h5 files (2) and then improve/change the call modifications file 
def get_X_Y(df):
    return df[df.columns[2:22]], df[df.columns[-1]]


def main(ko1, ko2, ko3, wt1, wt2, wt3, model_file):
    X, Y = get_X_Y(ko1)
    import pdb;pdb.set_trace()
    model = load_model(model_file)
    import pdb;pdb.set_trace()
    pred =  model.predict(data).flatten()
    inferred = np.zeros(len(pred), dtype=int)
    inferred[np.argwhere(pred >= 0.5)] = 1
    import pdb;pdb.set_trace()
    

if __name__ == "__main__":
    model_file = '/workspace/projects/nanopore/DeepMP/models/epinano_combined/error_model/' 
    bd = '/workspace/projects/nanopore/stockholm/EpiNano/novoa_features/yeast_epinano/'
    ko1 = get_id(pd.read_csv(os.path.join(bd, 'KO1/RRACH_5x_KO1.tsv'), sep='\t'))
    ko2 = get_id(pd.read_csv(os.path.join(bd, 'KO2/RRACH_5x_KO2.tsv'), sep='\t'))
    ko3 = get_id(pd.read_csv(os.path.join(bd, 'KO3/RRACH_5x_KO3.tsv'), sep='\t'))
    wt1 = get_id(pd.read_csv(os.path.join(bd, 'wildtype1/RRACH_5x_WT1.tsv'), sep='\t'))
    wt2 = get_id(pd.read_csv(os.path.join(bd, 'wildtype2/RRACH_5x_WT2.tsv'), sep='\t'))
    wt3 = get_id(pd.read_csv(os.path.join(bd, 'wildtype3/RRACH_5x_WT3.tsv'), sep='\t'))
    main(ko1, ko2, ko3, wt1, wt2, wt3, model_file)