from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics


def prepare_data_net():
    malware = pd.read_csv('data/net_malware.csv')
    benign = pd.read_csv('data/net_benign.csv')
    data = pd.concat([malware, benign], ignore_index=True)
    data = data.sort_values(by=['name'])
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=2020,
        test_size=0.3, stratify=y
    )

    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False,
                     random_state=2020).fit(X_train, y_train)
    sfm = SelectFromModel(lsvc, prefit=True)
    X_train = sfm.transform(X_train)
    X_test = sfm.transform(X_test)

    scaler = Normalizer()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def prepare_data_per():
    malware = pd.read_csv('data/per_malware.csv')
    benign = pd.read_csv('data/per_benign.csv')
    data = pd.concat([malware, benign], ignore_index=True)
    data = data.sort_values(by=['name'])
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=2020,
        test_size=0.3, stratify=y
    )

    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False,
                     random_state=2020).fit(X_train, y_train)
    sfm = SelectFromModel(lsvc, prefit=True)
    X_train = sfm.transform(X_train)
    X_test = sfm.transform(X_test)

    scaler = Normalizer()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def prepare_data_syscall():
    data = pd.read_csv('data/syscall.csv')
    data = data.sort_values(by=['name'])
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=2020,
        test_size=0.3, stratify=y
    )

    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False,
                     random_state=2020).fit(X_train, y_train)
    sfm = SelectFromModel(lsvc, prefit=True)
    X_train = sfm.transform(X_train)
    X_test = sfm.transform(X_test)

    scaler = Normalizer()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def load_model(net, per, sys):
    clf_net = pickle.load(open('model/net_' + net + '.sav', 'rb'))
    clf_per = pickle.load(open('model/per_' + per + '.sav', 'rb'))
    clf_sys = pickle.load(open('model/syscall_' + sys + '.sav', 'rb'))
    return clf_net, clf_per, clf_sys


X_train_net, y_train_net, X_test_net, y_test_net = prepare_data_net()
X_train_per, y_train_per, X_test_per, y_test_per = prepare_data_per()
X_train_sys, y_train_sys, X_test_sys, y_test_sys = prepare_data_syscall()

names = ["SVM", "k-NN", "DecisionTree", "RandomForest"]
df = list()
for net in ['k-NN']:
    for per in ['RandomForest']:
        for sys in ['k-NN']:
            res = dict()
            res['ensemble'] = net + ' + ' + per + ' + ' + sys
            clf_net, clf_per, clf_sys = load_model(net, per, sys)
            # Weak classifier
            res_net = 100 * metrics.accuracy_score(
                y_test_net, clf_net.predict(X_test_net))
            res_per = 100 * metrics.accuracy_score(
                y_test_per, clf_per.predict(X_test_per))
            res_sys = 100 * metrics.accuracy_score(
                y_test_sys, clf_sys.predict(X_test_sys))
            res['net'] = '%.2f' % res_net
            res['per'] = '%.2f' % res_per
            res['sys'] = '%.2f' % res_sys

            # Soft Voting
            y_pred_net = clf_net.predict_proba(X_test_net)
            y_pred_per = clf_per.predict_proba(X_test_per)
            y_pred_sys = clf_sys.predict_proba(X_test_sys)
            y_pred = (y_pred_net + y_pred_sys + y_pred_per) / 3
            y_pred = np.argmax(y_pred, axis=1)
            y_pred[y_pred==0] = -1
            res_vote = 100 * metrics.accuracy_score(y_test_net, y_pred)
            res['vote'] = '%.2f' % res_vote

            y_true = y_test_net
            with open('log/report_ensemble.txt', 'w') as f:
                clf_report = metrics.classification_report(y_true, y_pred, digits=4)
                cnf_matrix = metrics.confusion_matrix(y_true, y_pred)
                TN, FP, FN, TP = cnf_matrix.ravel()
                TPR = TP / (TP + FN)
                FPR = FP / (FP + TN)
                f.write('Accuracy: %0.4f\n' %
                        metrics.accuracy_score(y_true, y_pred))
                f.write('ROC AUC: %0.4f\n' %
                        metrics.roc_auc_score(y_true, y_pred))
                f.write('TPR: %0.4f\nFPR: %0.4f\n' % (TPR, FPR))
                f.write('Classification report:\n' + str(clf_report) + '\n')
                f.write('Confusion matrix:\n' + str(cnf_matrix) + '\n')

            # Prepare data
            y_pred_train_net = clf_net.predict_proba(
                X_train_net)[:, 0].reshape(-1, 1)
            y_pred_train_per = clf_per.predict_proba(
                X_train_per)[:, 0].reshape(-1, 1)
            y_pred_train_sys = clf_sys.predict_proba(
                X_train_sys)[:, 0].reshape(-1, 1)
            y_pred_train = np.hstack(
                [y_pred_train_net, y_pred_train_sys, y_pred_train_per])

            y_pred_test = np.hstack([y_pred_net[:, 0].reshape(-1, 1),
                                     y_pred_sys[:, 0].reshape(-1, 1),
                                     y_pred_per[:, 0].reshape(-1, 1)])

            # Logisic Regression
            reg = LogisticRegression(random_state=2020)
            reg.fit(y_pred_train, y_train_sys)
            y_pred = reg.predict(y_pred_test)
            res_log = 100 * metrics.accuracy_score(y_test_net, y_pred)
            res['logistic reg'] = '%.2f' % res_log

            df.append(res)

header = ['ensemble', 'net', 'per', 'sys',
          'vote', 'logistic reg']
pd.DataFrame(df)[header].to_csv('result.csv')
