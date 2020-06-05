from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pickle
import logging

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def report(names, y_true, y_pred):
    with open('log/report_per.txt', 'w') as f:
        for name in names:
            clf_report = metrics.classification_report(
                y_true[name], y_pred[name], digits=4)
            cnf_matrix = metrics.confusion_matrix(y_true[name], y_pred[name])
            TN, FP, FN, TP = cnf_matrix.ravel()
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            f.write(str(name) + ':\n')
            f.write('Accuracy: %0.4f\n' %
                    metrics.accuracy_score(y_true[name], y_pred[name]))
            f.write('ROC AUC: %0.4f\n' %
                    metrics.roc_auc_score(y_true[name], y_pred[name]))
            f.write('TPR: %0.4f\nFPR: %0.4f\n' % (TPR, FPR))
            f.write('Classification report:\n' + str(clf_report) + '\n')
            f.write('Confusion matrix:\n' + str(cnf_matrix) + '\n')
            f.write(64*'-'+'\n')


def prepare_data(args):
    malware = pd.read_csv('data/per_malware.csv')
    benign = pd.read_csv('data/per_benign.csv')
    data = pd.concat([malware, benign], ignore_index=True)
    data = data.sort_values(by=['name'])
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=args.seed,
        test_size=0.3, stratify=y
    )

    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False,
                     random_state=args.seed).fit(X_train, y_train)
    sfm = SelectFromModel(lsvc, prefit=True)
    X_train = sfm.transform(X_train)
    X_test = sfm.transform(X_test)

    scaler = Normalizer()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def main():
    parser = argparse.ArgumentParser(description='Machine Learning')
    parser.add_argument('-s', '--seed', type=int, default=2020,
                        metavar='S', help='random seed (default: 2020)')
    parser.add_argument('--test', action='store_true',
                        default=False, help='training or testing mode')
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%H:%M:%S.')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    y_true = {}
    y_pred = {}
    names = ["NaiveBayes", "SVM", "k-NN",
             "DecisionTree", "RandomForest"]
    classifiers = [
        GaussianNB(),
        SVC(kernel='rbf', probability=True),
        KNeighborsClassifier(n_jobs=-1),
        DecisionTreeClassifier(random_state=args.seed),
        RandomForestClassifier(random_state=args.seed, n_jobs=-1)
    ]
    hyperparam = [
        {},
        {'C': np.logspace(-3, 3, 7), 'gamma': ['scale', 'auto']},
        {'n_neighbors': [5, 50, 500], 'weights': [
            'uniform', 'distance'], 'algorithm': ['ball_tree', 'kd_tree', 'brute']},
        {'criterion': ['gini', 'entropy'], 'splitter': [
            'best', 'random'], 'max_features': ['sqrt', 'log2', None]},
        {'n_estimators': [10, 100, 1000], 'criterion': [
            'gini', 'entropy'], 'max_features': ['sqrt', 'log2', None]},
    ]

    X_train, y_train, X_test, y_test = prepare_data(args)
    logger.info(str(X_train.shape) + ' ' + str(X_test.shape))

    for name, est, hyper in zip(names, classifiers, hyperparam):
        logger.info('Classifier: ' + name)
        if not args.test:
            clf = GridSearchCV(est, hyper, cv=5, n_jobs=-1)
            clf.fit(X_train, y_train)
            logger.info('Score: %0.4f' % clf.score(X_train, y_train))
            y_true[name],  y_pred[name] = y_test, clf.predict(X_test)
            logger.info('Test accuracy: %0.4f' %
                        metrics.accuracy_score(y_true[name], y_pred[name]))
            pickle.dump(clf, open('model/per_' + str(name) + '.sav', 'wb'))
        else:
            clf = pickle.load(open('model/per_' + str(name) + '.sav', 'rb'))
            y_true[name], y_pred[name] = y_test, clf.predict(X_test)

    report(names, y_true, y_pred)


if __name__ == "__main__":
    main()
