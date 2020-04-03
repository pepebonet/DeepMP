#!/usr/bin/env python3
import os
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


def get_training_test_data(df):
    X = df[df.columns[:-1]]
    Y = df[df.columns[-1]]

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.1, random_state=0
    )

    X_train['label'] = Y_train
    X_val['label'] = Y_val

    return X_train, X_val


def save_files(train, val, output):
    train.to_csv(os.path.join(output, 'train_errors_rep2.csv'), index=None)
    val.to_csv(os.path.join(output, 'test_errors_rep2.csv'), index=None)


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
    train, val = get_training_test_data(df)
    save_files(train, val, output)


if __name__ == "__main__":
    main()