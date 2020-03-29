import os
import click
import pickle
import collections
import numpy as np 
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
    df_tmp = df.dropna(subset=names).reset_index(drop=True) 

    return df_tmp.iloc[:,cols], df_tmp.iloc[:,mod_col], df_tmp.index.tolist()


def evaluate_on_test_data (test, predictions):
    return np.sum(test.values == predictions) * 100 / len(test.values)


def get_accuracy(predictions, Y_test, kernels):
    accuracies = {}
    for _, kn in enumerate(kernels):
        accuracies[kn] = evaluate_on_test_data(Y_test, predictions[kn][0][1])

        acc_sort = sorted (accuracies.items(), key = lambda kv:kv[1])
        best_kn = acc_sort[-1][0]; best_acc = acc_sort[-1][1]
        print ("Best accuracy {} %  obtained with kernel = {}".format(
            best_acc,best_kn))
        
        del accuracies[best_kn]
        for k, v in accuracies.items ():
            print (" {} % accuracy obtained with kernel = {}".format(v,k))


#TODO Fix & Delete please (Get rid of the hardcoding)
def get_data_structure(row):
    import statistics
    listtt = [float(x) for x in row[0].split(',')]
    return statistics.mean(listtt)
#TODO Fix & Delete please (Get rid of the hardcoding)
def arrange_data(X, Y, X_test):
    X = X.apply(lambda row: get_data_structure(row), axis=1)
    Y.loc[0] = 0
    X_test = X_test.apply(lambda row: get_data_structure(row), axis=1)
    
    return X.values.reshape(1000, 1), Y.values, X_test.values.reshape(1000,1)


#Function still a bit ugly
def svm_prediction(kernels, X_train, Y_train, X_test, output):
    models_dict = collections.defaultdict(list)
    X, Y, X_test = arrange_data(X_train, Y_train, X_test)

    for _, kn in enumerate(kernels):
        model = svm.SVC(kernel=kn, probability=True)
        model_fit = model.fit (X, Y)
        predictions = model_fit.predict(X_test)
        models_dict[kn].append((model, predictions))

    return models_dict, X_test


def save_model(models, output, kernels):
    for _, kn in enumerate(kernels):
        out_model = os.path.join(output, '{}.model.dump'.format(kn))
        pickle.dump(models[kn][0][0], open(out_model,'wb'))


#TODO <JB> Needs work 
def save_predictions_to_file(models, output, kernels, X_test):
    for _, kn in enumerate(kernels):
        out_file = open(os.path.join(output, 'kernel.{}.csv'.format(kn)), 'w')

        for t in range (len(X_test)):
            original_line = ",".join(map(str, X_test[t]))
            dist = map(str,models[kn][0][0].decision_function([X_test[t]]))
            probM, probU = map(str,models[kn][0][0].predict_proba([X_test[t]])[0])
            print(original_line + ',' + str(X_test[t][0]) +',' 
                + ",".join (dist) + ',' +  probM + ',' + probU, file=out_file)

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
    
    predictions, X_test = svm_prediction(kernel, X_train, Y_train, X_test, output)

    if accuracy_est:
        get_accuracy(predictions, Y_test, kernel)

    if output:
        save_model(predictions, output, kernel)
        save_predictions_to_file(predictions, output, kernel, X_test)

    
if __name__ == '__main__':
    main()
