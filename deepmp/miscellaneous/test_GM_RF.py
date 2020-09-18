#!/usr/bin/envs python3
import os
import h5py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support 


def random_forest(X_train, Y_train, X_test, Y_test):
    rf = RandomForestRegressor(n_estimators = 100, n_jobs=40, random_state = 42)
    rf.fit(X_train, Y_train)
    predictions = rf.predict(X_test)
    return metrics(predictions, Y_test)


def gradient_boosting(X_train, Y_train, X_test, Y_test):
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.01)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    return metrics(predictions, Y_test)


def load_error_data(file):

    with h5py.File(file, 'r') as hf:
        X = hf['err_X'][:]
        Y = hf['err_Y'][:]

    return X, Y


def metrics(predictions, Y_test):
    inferred = np.zeros(len(Y_test), dtype=int)
    inferred[np.argwhere(predictions >= 0.5)] = 1

    precision, recall, f_score, _ = precision_recall_fscore_support(
        Y_test, inferred, average='binary'
    )

    return precision, recall, f_score


def main():
    base_dir = '/home/jbonet/Desktop/features_deepmp/ecoli/error_features/'
    X_train, Y_train = load_error_data(
        os.path.join(base_dir, 'train_err.h5')
    )
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])

    X_test, Y_test = load_error_data(
        os.path.join(base_dir, 'test.h5')
    )
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])
    print('Training random forest classifier...')
    p_rf, r_rf, f_rf = random_forest(X_train, Y_train, X_test, Y_test)
    print(p_rf, r_rf, f_rf)
    print('Training random gradient boosting machine...')
    p_gb, r_gb, f_gb = gradient_boosting(X_train, Y_train, X_test, Y_test)
    print(p_gb, r_gb, f_gb)

    import pdb;pdb.set_trace()  


if __name__ == "__main__":
    main()