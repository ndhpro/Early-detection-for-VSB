import itertools
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

# Loading dataset
malware = pd.read_csv('data/per_malware.csv')
benign = pd.read_csv('data/per_benign.csv')
data = pd.concat([malware, benign], ignore_index=True)
mal_list = pd.read_csv(
    'list_malware.csv', header=None).values[:-1].reshape(-1)
beg_list = pd.read_csv(
    'list_benign.csv', header=None).values[:-1].reshape(-1)

t_mal = int(len(mal_list) * 0.7)
t_beg = int(len(beg_list) * 0.7)
X_train = data[data['name'].isin(
    mal_list[:t_mal]) | data['name'].isin(beg_list[:t_beg])]
y_train = X_train.values[:, -1].astype('int')
X_train = X_train.values[:, 1:-1]

X_test = data[data['name'].isin(
    mal_list[t_mal:]) | data['name'].isin(beg_list[t_beg:])]
y_test = X_test.values[:, -1].astype('int')
X_test = X_test.values[:, 1:-1]

X_train = np.array(X_train.reshape(-1, 20, 20), dtype='Float32')
X_test = np.array(X_test.reshape(-1, 20, 20), dtype='Float32')
y_train = to_categorical(y_train, num_classes=2)
y_true = y_test
y_test = to_categorical(y_test, num_classes=2)

model = Sequential()
model.add(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2, input_shape=(20, 20)))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())

batch_size = 1024
checkpoint = ModelCheckpoint(
    'model/per_lstm.h5', save_best_only=True, save_weights_only=True)
history = model.fit(X_train, y_train, epochs=300, callbacks=[checkpoint],
                    batch_size=batch_size, verbose=2, validation_split=1/5)

model.load_weights('model/per_lstm.h5')

y_pred = model.predict(X_test, verbose=1, batch_size=batch_size)
y_pred = np.argmax(y_pred, axis=1)

with open('log/report_per.txt', 'w') as f:
    clf_report = metrics.classification_report(
        y_true, y_pred, digits=4)
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

# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r',
           label="validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'],
           color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.savefig('log/per_history.png')

# Drawing ROC curve
plt.figure()
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
auc = metrics.roc_auc_score(y_true, y_pred)
print(auc)
plt.plot(fpr, tpr, color='blue', label='AUC = %0.4f' % (auc))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.savefig('log/roc_per.png')
