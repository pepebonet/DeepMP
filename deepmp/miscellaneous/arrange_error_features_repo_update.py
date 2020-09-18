#!/usr/bin/env python3
import os
import h5py
import click
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(modified_features, unmodified_features):
    treat = pd.read_csv(modified_features)
    untreat = pd.read_csv(unmodified_features)
    return pd.concat([treat, untreat])


def get_labels_svm(df):
    mod = df[df['label'] == 1]; mod['label'] = 'mod'
    unm = df[df['label'] == 0]; unm['label'] = 'unm'
    return pd.concat([mod, unm])


def arrange_and_save_h5(X_train, Y_train, X_val, 
    Y_val, X_test, Y_test, output):
    X_train = X_train[X_train.columns[5:]].drop(
        columns=['ins1', 'ins2', 'ins3', 'ins4', 'ins5', 'label']
    ).values
    X_test = X_test[X_test.columns[5:]].drop(
        columns=['ins1', 'ins2', 'ins3', 'ins4', 'ins5', 'label']
    ).values
    X_val = X_val[X_val.columns[5:]].drop(
        columns=['ins1', 'ins2', 'ins3', 'ins4', 'ins5', 'label']
    ).values
    
    save_output(X_train, Y_train, output, X_train.shape[1], 'train')
    save_output(X_test, Y_test, output, X_test.shape[1], 'test')
    save_output(X_val, Y_val, output, X_val.shape[1], 'val')


def get_training_test_data(df, output):
    
    X = df[df.columns[:-1]]
    Y = df[df.columns[-1]]

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.05, random_state=0
    )

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_train, Y_train, test_size=X_val.shape[0], random_state=0
    )

    save_dataframes(X_train, Y_train, output, 'train')
    save_dataframes(X_test, Y_test, output, 'test')
    save_dataframes(X_val, Y_val, output, 'val')

    arrange_and_save_h5(X_train, Y_train.values, X_val, 
        Y_val.values, X_test, Y_test.values, output) 
    

def save_output(X, Y, output, feat, file):

    file_name = os.path.join(output, '{}_err.h5'.format(file))

    with h5py.File(file_name, 'a') as hf:
        hf.create_dataset("err_X", data=X.reshape(X.shape[0], feat, 1))
        hf.create_dataset("err_Y", data=Y)


def save_dataframes(feat, labels, output, file):
    feat['label'] = labels
    mod = feat[feat['label'] == 1].drop(
        columns=['label', 'ins1', 'ins2', 'ins3', 'ins4', 'ins5'])
    unm = feat[feat['label'] == 0].drop(
        columns=['label', 'ins1', 'ins2', 'ins3', 'ins4', 'ins5'])
    mod.to_csv(os.path.join(
        output, '{}_errors_mod.csv'.format(file)), index=None
    )
    unm.to_csv(os.path.join(
        output, '{}_errors_unm.csv'.format(file)), index=None
    )


@click.command(short_help='Script to get the right format of the' 
    'features and split into training and test')
@click.option(
    '-mf', '--modified-features', required=True,
    help='Features of the modified samples'
)
@click.option(
    '-umf', '--unmodified-features', required=True,
    help='Features of the unmodified samples'
)
@click.option(
    '-svm', '--svm-labelling', default=False,
    help='Put svm labelling instead of binary'
)
@click.option(
    '-o', '--output', required=True,
    help='output path to save training and test files'
)
def main(modified_features, unmodified_features, svm_labelling, output):
    df = get_data(modified_features, unmodified_features)
    if svm_labelling:
        df = get_labels_svm(df)
    get_training_test_data(df, output)


if __name__ == "__main__":
    main()