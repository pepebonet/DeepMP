import os
import click
import pickle
import collections
import pandas as pd 
from collections import Counter

from sklearn import svm
from sklearn.metrics import accuracy_score

from deepmp import utils as ut


# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------

def get_data(features_model, cols, mod_col):
    df = pd.read_csv(features_model, sep='\t')
    names = list(df.columns[cols])

    df_tmp = df.dropna(subset=names)
    df_tmp = df_tmp.reset_index(drop=True) 

    return df_tmp.iloc[:,cols], df_tmp.iloc[:,mod_col], df_tmp.index.tolist()


def evaluate_on_test_data (test, predictions):
    correct_classifications = 0
    for i in range (len(test)):
        if predictions[i] == test[i]:
            correct_classifications += 1
    accuracy = correct_classifications * 100 / len (test)
    return accuracy

#TODO Fix & Delete please
def get_data_structure(row):
    import statistics
    listtt = [float(x) for x in row[0].split(',')]
    return statistics.mean(listtt)


def svm_prediction(kernels, X_train, Y_train, X_test, output):
    models_dict = collections.defaultdict(list)
    X_train = X_train.apply(lambda row: get_data_structure(row), axis=1)
    X_test = X_test.apply(lambda row: get_data_structure(row), axis=1)
    Y_train.loc[0] = 0
    for _, kn in enumerate(kernels):
        model = svm.SVC(kernel=kn, probability=True)
        model_fit = model.fit (X_train.values.reshape(1000, 1), Y_train.values)
        predictions = model_fit.predict(X_test.values.reshape(1000,1))
        models_dict[kn].append((model, predictions))

    return predictions


def get_accuracy(predictions, Y_test, kernels):
    accuracies = {}
    for _, kn in enumerate(kernels):
        accuracies[kn] = evaluate_on_test_data(Y_test, predictions)
        acc_sort = sorted (accuracies.items(), key = lambda kv:kv[1])
        best_kn  = acc_sort[-1][0]
        best_acc = acc_sort[-1][1]
        # best_prediciton = output + '.best-kernel.' + best_kn + '.accuracy' 
        print ("Best accuracy {} %  obtained with kernel = {}".format(best_acc,best_kn))
        del accuracies[best_kn]
        for k, v in accuracies.items ():
            print (" {} % accuracy obtained with kernel = {}".format(v,k))


def save_model(output):
    out_model = output + '.' + kn + '.model.dump'
    pickle.dump (model,open (out_model,'wb'))
    outh = open(output+'.kernel.' + kn + '.csv','w')

# ------------------------------------------------------------------------------
# Click
# ------------------------------------------------------------------------------

@click.command(short_help='SVM algorithm as baseline model')
@click.option(
    '-k', '--kernel', multiple=True, default=None, 
    help='kernel used for training SVM'
)
@click.option(
    '-t', '--train_features', default='', 
    help='Feature table to train the model'
)
@click.option(
    '-p', '--predict_features', required=True, 
    help='Feature table make predictions'
)
@click.option(
    '-cl', '--columns', default='4', 
    help='comma seperated column number(s) that contain features '
        'used for training and prediciton'
)
@click.option(
    '-msc', '--mod_status_col', default=-1, 
    help='column number containing the modification status information'
)
@click.option(
    '-ae', '--accuracy_est', default=False, 
    help='Whether to obtain accuracy of predictions'
)
@click.option(
    '-o', '--output', default='', 
    help='Output file extension'
)
def main(kernel, train_features, columns, predict_features, 
    mod_status_col, accuracy_est, output):
    cols = ut.arrange_columns(columns)
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']  if kernel is None else kernel

    if train_features:
        X_train, Y_train, index_train = get_data(
            train_features, cols, mod_status_col
        )

    if predict_features:
        X_test, Y_test, index_test = get_data(
            predict_features, cols, mod_status_col
        )
    
    predictions = svm_prediction(kernel, X_train, Y_train, X_test, output)

    if accuracy_est:
        get_accuracy(predictions, Y_test, kernel)

    
if __name__ == '__main__':
    main()
