from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pickle
import logging

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import metrics
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer


def draw_roc(names, colors, y_true, y_pred):
    for name, color in zip(names, colors):
        fpr, tpr, _ = metrics.roc_curve(y_true[name], y_pred[name])
        auc = metrics.roc_auc_score(y_true[name], y_pred[name])
        plt.plot(fpr, tpr, color=color,
                 label='%s ROC (area = %0.3f)' % (name, auc))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('log/roc_syscall.png')


def report(names, y_true, y_pred):
    with open('log/report_syscall.txt', 'w') as f:
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
    data = pd.read_csv('data/syscall.csv')
    mal_list = pd.read_csv(
        'list_malware.csv', header=None).values[:-1].reshape(-1)
    beg_list = pd.read_csv(
        'list_benign.csv', header=None).values[:-1].reshape(-1)

    X_train = data[data['name'].isin(
        mal_list[:448]) | data['name'].isin(beg_list[:232])]
    y_train = X_train.values[:, -1].astype('int')
    X_train = X_train.values[:, 1:-1]

    X_test = data[data['name'].isin(
        mal_list[448:]) | data['name'].isin(beg_list[232:])]
    y_test = X_test.values[:, -1].astype('int')
    X_test = X_test.values[:, 1:-1]
    
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
    names = ["SVM", "k-Nearest Neighbors",
             "Decision Tree", "Random Forest", "Bagging"]
    classifiers = [
        SVC(kernel='rbf'),
        KNeighborsClassifier(n_jobs=-1),
        DecisionTreeClassifier(random_state=args.seed),
        RandomForestClassifier(random_state=args.seed, n_jobs=-1),
        BaggingClassifier(random_state=args.seed, n_jobs=-1)
    ]
    hyperparam = [
        {'C': np.logspace(-3, 3, 7), 'gamma': ['scale', 'auto']},
        {'n_neighbors': [5, 50, 500], 'weights': [
            'uniform', 'distance'], 'algorithm': ['ball_tree', 'kd_tree', 'brute']},
        {'criterion': ['gini', 'entropy'], 'splitter': [
            'best', 'random'], 'max_features': ['sqrt', 'log2', None]},
        {'n_estimators': [10, 100, 1000], 'criterion': [
            'gini', 'entropy'], 'max_features': ['sqrt', 'log2', None]},
        {'n_estimators': [10, 100, 1000]}
    ]
    colors = ['blue', 'orange', 'green', 'red',
              'purple', 'brown', 'pink', 'gray']

    X_train, y_train, X_test, y_test = prepare_data(args)
    logger.info(str(X_train.shape) + ' ' + str(X_test.shape))

    for name, est, hyper in zip(names, classifiers, hyperparam):
        logger.info(name + '...')
        if not args.test:
            clf = GridSearchCV(est, hyper, cv=5, n_jobs=-1)
            clf.fit(X_train, y_train)
            y_true[name],  y_pred[name] = y_test, clf.predict(X_test)
            logger.info('___accuracy: %0.4f' %
                        metrics.accuracy_score(y_true[name], y_pred[name]))
            print(clf.best_estimator_)
            pickle.dump(clf, open('model/syscall_' + str(name) + '.sav', 'wb'))
        else:
            clf = pickle.load(open('model/syscall_' + str(name) + '.sav', 'rb'))
            y_true[name], y_pred[name] = y_test, clf.predict(X_test)

    report(names, y_true, y_pred)
    draw_roc(names, colors, y_true, y_pred)


if __name__ == "__main__":
    # prepare_data(None)
    main()
