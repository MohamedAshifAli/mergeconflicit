import os

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from keras import Sequential, Model
from keras.layers import *
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.utils import to_categorical
import tensorflow as tf
import warnings

from termcolor import colored

# from optimzr import SOA

warnings.filterwarnings('ignore')
# data = pd.read_csv("Datasets/reduced_data_1.csv")
# data1=pd.read_csv("Datasets/reduced_data_2.csv")
# data2=pd.read_csv("Datasets/reduced_data_3.csv")
# data3=pd.read_csv("Datasets/reduced_data_4.csv")
# #
# concatenated_data = pd.concat([data, data1, data2, data3])
# concatenated_data.to_csv("concatenated_data.csv", index=False)

# Step 1: Load the CSV file into a DataFrame
def oversample(X_train, y_train):
    ros = SMOTE(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

class Optimization:
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def main_weight_updation_optimization(self, curr_wei):
        print((colored("[INFO] Coyote Optimization \U0001F43A", 'magenta', on_color='on_grey')))
        model = SOA.SOA(curr_wei,self.model,self.x_test,self.y_test)
        best_position2, best_fitness2 = model.solve()
        return best_position2

    def main_update_hyperparameters(self):
        wei_to_train = self.model.get_weights()
        to_opt_1 = wei_to_train[0]
        re_to_opt_1 = to_opt_1.reshape(to_opt_1.shape[0] * to_opt_1.shape[1], to_opt_1.shape[2] * to_opt_1.shape[3])
        wei_to_train_1 = self.main_weight_updation_optimization(re_to_opt_1)
        to_opt_new = wei_to_train_1.reshape(to_opt_1.shape[0], to_opt_1.shape[1], to_opt_1.shape[2], to_opt_1.shape[3])
        wei_to_train[0] = to_opt_new
        self.model.set_weights(wei_to_train)
        return self.model

def main_est_perf_metrics(preds, y_test):

    confusion_mtx = confusion_matrix(y_test, preds)
    #cm =sum(mcm)
    """Total Counts of Confusion Matrix"""
    total = sum(sum(confusion_mtx))
    # """True Positive"""
    TP = confusion_mtx[0, 0]
    """False Positive"""
    FP = confusion_mtx[0, 1]
    """False Negative"""
    FN = confusion_mtx[1, 0]
    """True Negative"""
    TN = confusion_mtx[1, 1]
    """Accuracy Formula"""
    acc = (TP + TN) / total
    """Sensitivity Formula"""
    sen = TP / (FN + TP)
    """Specificity Formula"""
    spe = TN / (FP + TN)
    """Precision Formula"""
    pre = TP / (TP + FP)
    """Recall Formula"""
    rec = TP / (FN + TP)
    """F1 Score Formula"""
    f1_score = (2 * pre * rec) / (pre + rec)
    '''Critical Success Index '''
    CSI = TP/(TP+FN+FP)
    '''False Positivie Rate'''
    FPR = FP / (FP + TN)  # 1 - Specificity
    '''False Negative Rate'''
    FNR = FN / (TP + FN)  # 1 - Sensitivity
    '''Matthews Correlation Coefficient'''
    MCC = (TP * TN - FP * FN) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    '''Negative Predictive Value'''
    NPV = TN / (TN + FN)  # negative predictive value
    '''Positive Predictive Value'''
    PPV = TP / (TP + FP)  # Positive Predictive Value

    return [acc, sen, spe, pre,rec,f1_score,CSI,FPR,FNR,MCC,PPV,NPV]


def preprocess():
    concatenated_data = pd.read_csv("Datasets/reduced_data_4.csv")[576500:577500]
    concatenated_data.drop(columns=['pkSeqID'], inplace=True)
    features = concatenated_data.drop(columns=['attack'])
    # labels = concatenated_data['attack']

    labels = concatenated_data['attack'].values

    label_encoder = LabelEncoder()
    for column in features.columns:
        if features[column].dtype == 'object':
            try:
                features[column] = label_encoder.fit_transform(features[column])
            except:
                features[column]=features[column].astype("str")
                features[column] = label_encoder.fit_transform(features[column])
    features,labels = oversample(features,labels)


    np.save("Features\\features.npy", features)
    np.save("Features\\labels.npy", labels)
    return features,labels
def split_data(features, labels,  train_size):
    total_data = len(features)
    indices = np.arange(total_data)
    np.random.shuffle(indices)

    split_index = int(total_data * (1 - train_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    xtrain = features[train_indices]
    xtest = features[test_indices]
    ytrain = labels[train_indices]
    ytest = labels[test_indices]

    return xtrain, xtest, ytrain, ytest
# def split_data1(features, labels,  train_size):
#     total_data = len(features)
#     indices = np.arange(total_data)
#     np.random.shuffle(indices)
#     sorted_labels = labels[indices].copy()
#
#     split_index = int(total_data * (1 - train_size))
#     train_indices = indices[:split_index]
#     test_indices = indices[split_index:]
#     xtrain = features[train_indices]
#     xtest = features[test_indices]
#     ytrain = sorted_labels[:split_index]
#     ytest = sorted_labels[split_index:]
#
#     return xtrain, ytrain, xtest, ytest

#comparative models
def RNN(xtrain,ytrain,xtest, ytest,epochs):

    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1] // 4, 4)
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1] // 4, 4)

    model = Sequential()
    model.add(SimpleRNN(64, input_shape=xtrain.shape[1:], activation="tanh"))
    model.add(Dense(ytrain.shape[1], activation="softmax"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(
        '\033[46m' + '\033[30m' + "________________________RNN CNN Model prepared...__________________________________" + '\x1b[0m')

    model.fit(xtrain, ytrain, batch_size=32, epochs=epochs, validation_split=0.2)
    Y_pred = model.predict(xtest)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(ytest, axis=1)
    return Y_pred, Y_true

def SVM_Model(xtrain,ytrain,xtest, ytest):

    # xtrain = xtrain [:700]
    # ytrain = ytrain[:700]
    xtrain,ytrain = oversample(xtrain,ytrain)
    # xtest,ytest = oversample(xtest,ytest)


    print('\33[45m' + '\33[37m' + "[INFO] -------Loading SVM---------" + '\x1b[0m')
    model = SVC(verbose=1)
    print('\33[106m' + '\33[40m' + "[INFO] -------Training SVM---------" + '\x1b[0m')
    # model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

    try:
        model.fit(xtrain, ytrain)
    except:
        print("ss")
    print(colored("[INFO] -------Testing SVM---------","magenta"))
    Y_pred = model.predict(xtest)
    # Y_pred = np.argmax(Y_pred)
    Y_true = np.array(ytest)
    return Y_pred, Y_true

def Bi_LSTM_CNN(xtrain,ytrain,xtest, ytest,epochs):
    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1, 1)
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1, 1)
    input_layer =  Input(shape=(xtrain.shape[1], xtrain.shape[2], xtrain.shape[3]))

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPool2D(1, 1)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(16, (3, 3), activation='relu', padding="same")(x)
    x = MaxPool2D(1, 1)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
    x = MaxPool2D(1, 1)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding="same")(x)
    x = MaxPool2D(1, 1)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(1, 1)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(256, (2, 2), activation='relu', padding='same')(x)
    x = MaxPool2D(1, 1)(x)
    x = Reshape((x.shape[1], x.shape[3]))(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(ytrain.shape[1], activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

    model.fit(xtrain, ytrain, batch_size=32, epochs=epochs)
    Y_pred = model.predict(xtest)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(ytest, axis=1)
    return Y_pred, Y_true

def DNN_Model(xtrain, ytrain, xtest, ytest, epochs):


    print('\33[45m' + '\33[37m' + "[INFO] -------Loading DNN---------" + '\x1b[0m')
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=300, activation="relu"))
    model.add(tf.keras.layers.Dense(units=200, activation="relu"))
    model.add(tf.keras.layers.Dense(units=100, activation="relu"))
    model.add(tf.keras.layers.Dense(units=50, activation="relu"))
    model.add(tf.keras.layers.Dense(units=ytrain.shape[1], activation="softmax"))
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    print('\33[106m' + '\33[40m' + "[INFO] -------Training DNN---------" + '\x1b[0m')
    model.fit(xtrain, ytrain, epochs=epochs, batch_size=64, verbose=1)
    print(colored("[INFO] -------Testing DNN---------", "magenta"))
    Y_pred = model.predict(xtest)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(ytest, axis=1)
    return Y_pred, Y_true

# Define your CNN model
def Deep_CNN(xtrain,ytrain,xtest, ytest,epochs):
    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1] // 2, 2, 1)
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1] // 2, 2, 1)

    model = tf.keras.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=xtrain.shape[1:],padding='SAME' ))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',padding='SAME'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(BatchNormalization())

    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu',padding='SAME'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(BatchNormalization())

    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu',padding='SAME'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',padding='SAME'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(ytrain.shape[1], activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(
        '\033[46m' + '\033[30m' + "________________________Deep CNN Model prepared...__________________________________" + '\x1b[0m')

    model.fit(xtrain, ytrain, batch_size=32, epochs=epochs, validation_split=0.2)
    # if option == 0:
    #     model = model
    # else:
    #     op = Optimization(model, xtest, ytest)
    #     model = op.main_update_hyperparameters()

    Y_pred = model.predict(xtest)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(ytest, axis=1)
    return Y_pred, Y_true

def TP_Analysis(features,labels):

    epochs = [5,10,30,35,50]
    tr =[0.3,0.7, 0.8]
    options = [0, 1]

    COM_A = []
    COM_B = []
    COM_C = []
    COM_D = []
    COM_E = []
    COM_F = []
    COM_G = []
    COM_H = []
    COM_I = []

    for p in range(len(tr)):
        print(  '\033[46m' + '\033[30m' + "Training Percentage and Testing Percentage : " + str(tr[p] * 100) + " and " + str(
                100 - (tr[p] * 100)) + '\x1b[0m')

        xtrain, xtest, ytrain, ytest = split_data(features, labels, train_size=tr[p])
        y1train = to_categorical(ytrain)
        y1test = to_categorical(ytest)
        print(
            '\033[46m' + '\033[30m' + "------------------------------MODEL TRAINING SECTION---------------------------------------" + '\x1b[0m')
        Y_pred1, Y_true1 = DNN_Model(xtrain, y1train, xtest, y1test, epochs[0])
        Y_pred2, Y_true2 = Bi_LSTM_CNN(xtrain, y1train, xtest, y1test, epochs[0])
        Y_pred3, Y_true3 = RNN(xtrain, y1train, xtest, y1test, epochs[0])
        # xtrain, xtest, ytrain, ytest = split_data(features, labels, train_size=tr[p])
        Y_pred4, Y_true4 = SVM_Model(xtrain, ytrain, xtest, ytest)
        Y_pred5, Y_true5 = Deep_CNN(xtrain, y1train, xtest, y1test, epochs[0])
        Y_pred6, Y_true6 = Deep_CNN(xtrain, y1train, xtest, y1test, epochs[1])
        Y_pred7, Y_true7 = Deep_CNN(xtrain, y1train, xtest, y1test, epochs[2])
        Y_pred8, Y_true8 = Deep_CNN(xtrain, y1train, xtest, y1test, epochs[3])
        Y_pred9, Y_true9 = Deep_CNN(xtrain, y1train, xtest, y1test, epochs[4])

        [ACC1, SEN1, SPE1, PRE1, REC1, FSC1, CSI1, FPR1, FNR1, MCC1, PPV1, NPV1] = main_est_perf_metrics(Y_pred1,
                                                                                                         Y_true1)
        [ACC2, SEN2, SPE2, PRE2, REC2, FSC2, CSI2, FPR2, FNR2, MCC2, PPV2, NPV2] = main_est_perf_metrics(Y_pred2,
                                                                                                         Y_true2)
        [ACC3, SEN3, SPE3, PRE3, REC3, FSC3, CSI3, FPR3, FNR3, MCC3, PPV3, NPV3] = main_est_perf_metrics(Y_pred3,
                                                                                                         Y_true3)
        [ACC4, SEN4, SPE4, PRE4, REC4, FSC4, CSI4, FPR4, FNR4, MCC4, PPV4, NPV4] = main_est_perf_metrics(Y_pred4,
                                                                                                         Y_true4)
        [ACC5, SEN5, SPE5, PRE5, REC5, FSC5, CSI5, FPR5, FNR5, MCC5, PPV5, NPV5] = main_est_perf_metrics(Y_pred5,
                                                                                                         Y_true5)

        [ACC6, SEN6, SPE6, PRE6, REC6, FSC6, CSI6, FPR6, FNR6, MCC6, PPV6, NPV6] = main_est_perf_metrics(Y_pred6,Y_true6)

        [ACC7, SEN7, SPE7, PRE7, REC7, FSC7, CSI7, FPR7, FNR7, MCC7, PPV7, NPV7] = main_est_perf_metrics(Y_pred7,  Y_true7)

        [ACC8, SEN8, SPE8, PRE8, REC8, FSC8, CSI8, FPR8, FNR8, MCC8, PPV8, NPV8] = main_est_perf_metrics(Y_pred8,  Y_true8)

        [ACC9, SEN9, SPE9, PRE9, REC9, FSC9, CSI9, FPR9, FNR9, MCC9, PPV9, NPV9] = main_est_perf_metrics(Y_pred9,    Y_true9)

        COM_A.append([ACC1, SEN1, SPE1, PRE1, REC1, FSC1, CSI1, FPR1, FNR1, MCC1, PPV1, NPV1])
        COM_B.append([ACC2, SEN2, SPE2, PRE2, REC2, FSC2, CSI2, FPR2, FNR2, MCC2, PPV2, NPV2])
        COM_C.append([ACC3, SEN3, SPE3, PRE3, REC3, FSC3, CSI3, FPR3, FNR3, MCC3, PPV3, NPV3])
        COM_D.append([ACC4, SEN4, SPE4, PRE4, REC4, FSC4, CSI4, FPR4, FNR4, MCC4, PPV4, NPV4])
        COM_E.append([ACC5, SEN5, SPE5, PRE5, REC5, FSC5, CSI5, FPR5, FNR5, MCC5, PPV5, NPV5])
        COM_F.append([ACC6, SEN6, SPE6, PRE6, REC6, FSC6, CSI6, FPR6, FNR6, MCC6, PPV6, NPV6])
        COM_G.append([ACC7, SEN7, SPE7, PRE7, REC7, FSC7, CSI7, FPR7, FNR7, MCC7, PPV7, NPV7])
        COM_H.append([ACC8, SEN8, SPE8, PRE8, REC8, FSC8, CSI8, FPR8, FNR8, MCC8, PPV8, NPV8])
        COM_I.append([ACC9, SEN9, SPE9, PRE9, REC9, FSC9, CSI9, FPR9, FNR9, MCC9, PPV9, NPV9])

    np.save('NPY\\COM_A.npy'.format(os.getcwd()), COM_A)
    np.save('NPY\\COM_B.npy'.format(os.getcwd()), COM_B)
    np.save('NPY\\COM_C.npy'.format(os.getcwd()), COM_C)
    np.save('NPY\\COM_D.npy'.format(os.getcwd()), COM_D)
    np.save('NPY\\COM_E.npy'.format(os.getcwd()), COM_E)
    np.save('NPY\\COM_F.npy'.format(os.getcwd()), COM_F)
    np.save('NPY\\COM_G.npy'.format(os.getcwd()), COM_G)
    np.save('NPY\\COM_H.npy'.format(os.getcwd()), COM_H)
    np.save('NPY\\COM_I.npy'.format(os.getcwd()), COM_I)

def parameter(acc, sen, spe, pre, rec, fsc,csi,fpr,fnr,mcc,ppv,npv):
    acc = np.mean(acc)
    sen = np.mean(sen)
    spe = np.mean(spe)
    pre = np.mean(pre)
    rec = np.mean(rec)
    fsc = np.mean(fsc)
    csi=np.mean(csi)
    fpr=np.mean(fpr)
    fnr=np.mean(fnr)
    mcc=np.mean(mcc)
    ppv=np.mean(ppv)
    npv=np.mean(npv)
    return [acc, sen, spe, pre, rec, fsc,csi,fpr,fnr,mcc,ppv,npv]

def  KF_Analysis(features,labels):

    options = [0, 1]
    # features, labels = oversample(features, labels)
    """K-fold cross-validation approach divides the input dataset into K groups of samples of equal sizes.
     These samples are called folds. For each learning set, the prediction function uses k-Cotton Leaf folds,
      and the rest of the folds are used for the test set. This approach is a very popular CV approach
      because it is easy to understand, and the output is less biased than other methods."""
    kr = [4, 6, 8, 10]
    """An epoch in machine learning means one complete pass of the training dataset through the algorithm.
         This epoch's number is an important hyperparameter for the algorithm. 
         It specifies the number of epochs or complete passes of the entire training dataset passing through the training or learning process of the algorithm"""
    # epochs =[2,3,4,5,1] # No. of Iterations
    # epochs = [10,20,30,40,50]
    epochs = [5, 10, 15,20,30 ]
    # epochs = [20, 40, 60, 80, 100]
    """A label represents an output value,
       while a feature is an input value that describes the characteristics of such labels in datasets"""
    features = np.nan_to_num(features, 0)
    options = [0, 1, 2, 3]
    """Normalization is a technique often applied as part of data preparation for machine learning. 
        The goal of normalization is to change the values of numeric columns in the dataset to use a common scale,
         without distorting differences in the ranges of values or losing information"""
    ## Normalization
    feat = features.astype(np.float32) / features.max()

    COM_A = []
    COM_B = []
    COM_C = []
    COM_D = []
    COM_E = []


    """The steps for k-fold cross-validation are:
            Split the input dataset into K groups
        For each group:
        Take one group as the reserve or test data set.
        Use remaining groups as the training dataset
        Fit the model on the training set and evaluate the performance of the model using the test set."""
    for w in range(len(kr)):
        print(kr[w])
        strtfdKFold = StratifiedKFold(n_splits=kr[w])
        kfold = strtfdKFold.split(feat, labels)
        acc1, sen1, spe1, pre1, rec1, fsc1,csi1,fpr1,fnr1,mcc1,ppv1,npv1 = [], [], [], [], [], [], [], [], [], [], [], []
        acc2, sen2, spe2, pre2, rec2, fsc2,csi2,fpr2,fnr2,mcc2,ppv2,npv2 = [], [], [], [], [], [], [], [], [], [], [], []
        acc3, sen3, spe3, pre3, rec3, fsc3,csi3,fpr3,fnr3,mcc3,ppv3,npv3 = [], [], [], [], [], [], [], [], [], [], [], []
        acc4, sen4, spe4, pre4, rec4, fsc4,csi4,fpr4,fnr4,mcc4,ppv4,npv4 = [], [], [], [], [], [], [], [], [], [], [], []
        acc5, sen5, spe5, pre5, rec5, fsc5,csi5,fpr5,fnr5,mcc5,ppv5,npv5= [], [], [], [], [], [], [], [], [], [], [], []

        for k, (train, test) in enumerate(kfold):
            if k==0:
                tr_data = feat[train, :]
                tr_data = tr_data[:, :]
                ytrain = labels[train]
                tst_data = feat[test, :]
                tst_data = tst_data[:, :]
                ytest = labels[test]
                xtrain = tr_data
                xtest = tst_data

                y1train = keras.utils.to_categorical(ytrain)
                y1test = keras.utils.to_categorical(ytest)

                print("------------------------------MODEL TRAINING SECTION---------------------------------------")

                Y_pred1, Y_true1 = Deep_CNN(xtrain, y1train, xtest, y1test, epochs[4])
                Y_pred2, Y_true2 = Deep_CNN(xtrain, y1train, xtest, y1test, epochs[4])
                Y_pred3, Y_true3 = Deep_CNN(xtrain, y1train, xtest, y1test, epochs[4])
                Y_pred4, Y_true4 = Deep_CNN(xtrain, y1train, xtest, y1test, epochs[4])
                Y_pred5, Y_true5 = Deep_CNN(xtrain, y1train, xtest, y1test, epochs[4])

                print(
                    "________________________Metrics Evaluated from Confusion Matrix__________________________________")
                [ACC1, SEN1, SPE1, PRE1, REC1, FSC1, CSI1, FPR1, FNR1, MCC1, PPV1, NPV1] = main_est_perf_metrics(
                    Y_pred1, Y_true1)
                [ACC2, SEN2, SPE2, PRE2, REC2, FSC2, CSI2, FPR2, FNR2, MCC2, PPV2, NPV2] = main_est_perf_metrics(
                    Y_pred2, Y_true2)
                [ACC3, SEN3, SPE3, PRE3, REC3, FSC3, CSI3, FPR3, FNR3, MCC3, PPV3, NPV3] = main_est_perf_metrics(
                    Y_pred3, Y_true3)
                [ACC4, SEN4, SPE4, PRE4, REC4, FSC4, CSI4, FPR4, FNR4, MCC4, PPV4, NPV4] = main_est_perf_metrics(
                    Y_pred4, Y_true4)
                [ACC5, SEN5, SPE5, PRE5, REC5, FSC5, CSI5, FPR5, FNR5, MCC5, PPV5, NPV5] = main_est_perf_metrics(
                    Y_pred5, Y_true5)



                acc1.append(ACC1)
                sen1.append(SEN1)
                spe1.append(SPE1)
                pre1.append(PRE1)
                rec1.append(REC1)
                fsc1.append(FSC1)
                acc2.append(ACC2)
                sen2.append(SEN2)
                spe2.append(SPE2)
                pre2.append(PRE2)
                rec2.append(REC2)
                fsc2.append(FSC2)
                acc3.append(ACC3)
                sen3.append(SEN3)
                spe3.append(SPE3)
                pre3.append(PRE3)
                rec3.append(REC3)
                fsc3.append(FSC3)
                acc4.append(ACC4)
                sen4.append(SEN4)
                spe4.append(SPE4)
                pre4.append(PRE4)
                rec4.append(REC4)
                fsc4.append(FSC4)
                acc5.append(ACC5)
                sen5.append(SEN5)
                spe5.append(SPE5)
                pre5.append(PRE5)
                rec5.append(REC5)
                fsc5.append(FSC5)
                csi1.append(CSI1)
                fpr1.append(FPR1)
                fnr1.append(FNR1)
                csi2.append(CSI2)
                fpr2.append(FPR2)
                fnr2.append(FNR2)
                csi3.append(CSI3)
                fpr3.append(FPR3)
                fnr3.append(FNR3)
                csi4.append(CSI4)
                fpr4.append(FPR4)
                fnr4.append(FNR4)
                csi5.append(CSI5)
                fpr5.append(FPR5)
                fnr5.append(FNR5)
                csi1.append(CSI1)
                fpr1.append(FPR1)
                fnr1.append(FNR1)
                csi2.append(CSI2)
                fpr2.append(FPR2)
                fnr2.append(FNR2)
                csi3.append(CSI3)
                fpr3.append(FPR3)
                fnr3.append(FNR3)
                csi4.append(CSI4)
                fpr4.append(FPR4)
                fnr4.append(FNR4)
                csi5.append(CSI5)
                fpr5.append(FPR5)
                fnr5.append(FNR5)
                mcc1.append(MCC1)
                ppv1.append(PPV1)
                npv1.append(NPV1)
                mcc2.append(MCC2)
                ppv2.append(PPV2)
                npv2.append(NPV2)
                mcc3.append(MCC3)
                ppv3.append(PPV3)
                npv3.append(NPV3)
                mcc4.append(MCC4)
                ppv4.append(PPV4)
                npv4.append(NPV4)
                mcc5.append(MCC5)
                ppv5.append(PPV5)
                npv5.append(NPV5)
            [ACC_1, SEN_1, SPE_1, PRE_1, REC_1, FSC_1, CSI_1, FPR_1, FNR_1, MCC_1, PPV_1, NPV_1] =parameter(acc1, sen1, spe1, pre1,rec1, fsc1,csi1, fpr1,fnr1, mcc1, ppv1, npv1)
            [ACC_2, SEN_2, SPE_2, PRE_2, REC_2, FSC_2, CSI_2, FPR_2, FNR_2, MCC_2, PPV_2, NPV_2] = parameter(acc2, sen2,spe2, pre2,rec2, fsc2,csi2, fpr2,fnr2, mcc2,ppv2, npv2)
            [ACC_3, SEN_3, SPE_3, PRE_3, REC_3, FSC_3, CSI_3, FPR_3, FNR_3, MCC_3, PPV_3, NPV_3] = parameter(acc3, sen3,spe3, pre3,rec3, fsc3,csi3, fpr3,fnr3, mcc3,ppv3, npv3)
            [ACC_4, SEN_4, SPE_4, PRE_4, REC_4, FSC_4, CSI_4, FPR_4, FNR_4, MCC_4, PPV_4, NPV_4] = parameter(acc4, sen4, spe4, pre4,rec4, fsc4,csi4, fpr4, fnr4, mcc4,ppv4, npv4)
            [ACC_5, SEN_5, SPE_5, PRE_5, REC_5, FSC_5, CSI_5, FPR_5, FNR_5, MCC_5, PPV_5, NPV_5] = parameter(acc5, sen5,  spe5, pre5,rec5, fsc5, csi5, fpr5,fnr5, mcc5,ppv5, npv5)


            COM_A.append([ACC_1, SEN_1, SPE_1, PRE_1, REC_1, FSC_1, CSI_1, FPR_1, FNR_1, MCC_1, PPV_1, NPV_1])
            COM_B.append([ACC_2, SEN_2, SPE_2, PRE_2, REC_2, FSC_2, CSI_2, FPR_2, FNR_2, MCC_2, PPV_2, NPV_2])
            COM_C.append([ACC_3, SEN_3, SPE_3, PRE_3, REC_3, FSC_3, CSI_3, FPR_3, FNR_3, MCC_3, PPV_3, NPV_3])
            COM_D.append([ACC_4, SEN_4, SPE_4, PRE_4, REC_4, FSC_4, CSI_4, FPR_4, FNR_4, MCC_4, PPV_4, NPV_4])
            COM_E.append([ACC_5, SEN_5, SPE_5, PRE_5, REC_5, FSC_5, CSI_5, FPR_5, FNR_5, MCC_5, PPV_5, NPV_5])

        np.save('NPY1\\COM_A.npy'.format(os.getcwd()), COM_A)
        np.save('NPY1\\COM_B.npy'.format(os.getcwd()), COM_B)
        np.save('NPY1\\COM_C.npy'.format(os.getcwd()), COM_C)
        np.save('NPY1\\COM_D.npy'.format(os.getcwd()), COM_D)
        np.save('NPY1\\COM_E.npy'.format(os.getcwd()), COM_E)


def Complete_Figure_com(perf, val, str_1, xlab, ylab, name):
    perf = perf * 100
    a = perf[:, 0]
    b = perf[:, 1]
    c = perf[:, 2]

    dict = {'4': a, '6': b, '8': c}
    df = pd.DataFrame(dict, index=[str_1])

    df.to_csv('Results_P1\\KF\\Comp_Analysis\\' + '_' + str(val) + str(name) + '_' + 'Graph.csv')

    df1 = {
        'No.of.records': ['4', '4', '4', '6', '6', '6', '8', '8', '8'],
        'Accuracy(%)': [
            perf[0, 0], perf[1, 0], perf[2, 0],
            perf[0, 1], perf[1, 1], perf[2, 1],
            perf[0, 2], perf[1, 2], perf[2, 2]
        ],
        'Legend': [
            str_1[0], str_1[1], str_1[2],
            str_1[0], str_1[1], str_1[2],
            str_1[0], str_1[1], str_1[2]
        ]
    }

    plt.figure()
    sns.set(font_scale=0.8)
    sns.set_style("whitegrid")
    sns.barplot(x='No.of.records', y='Accuracy(%)', hue='Legend', palette=['#6600ff', '#ff9900', '#cc00ff'], data=df1)
    plt.legend(ncol=2, loc='lower center', fontsize=15, prop={"weight": "bold"})
    plt.xlabel(xlab, fontsize=15, weight='bold')
    plt.ylabel(ylab, fontsize=15, weight='bold')
    plt.savefig('Results_P1\\KF\\Comp_Analysis\\' + str(val) + '_' + str(name) + 'Graph.png', dpi=800)
    plt.show(block=False)
    plt.clf()
#
# def Complete_Figure_com(perf, val, str_1, xlab, ylab,name):
#     perf = perf * 100
#     a = perf[:, 0]
#     b = perf[:, 1]
#     c = perf[:, 2]
#     d = perf[:, 3]
#     e= perf[:,4]
#     dict = {'40': a, '50': b, '60': c, '70': d,'80':e}
#     df = pd.DataFrame(dict, index=[str_1])
#     df.to_csv('Results_P1\\TP\\Comp_Analysis\\' + '_' + str(val) + str(name) +'_' + 'Graph.csv')
#     df1 = {
#         'No.of.records': ['40', '40', '40', '40', '40', '40', '40', '40',
#                           '50', '50', '50', '50', '50', '50', '50', '50',
#                           '60', '60', '60', '60', '60', '60', '60', '60',
#                           '70', '70', '70', '70', '70', '70', '70', '70',
#                           '80', '80', '80', '80', '80', '80', '80', '80'],
#         'Accuracy(%)': [perf[0, 0], perf[1, 0], perf[2, 0], perf[3, 0], perf[4, 0], perf[5, 0], perf[6, 0], perf[7, 0]
#             , perf[0, 1], perf[1, 1], perf[2, 1], perf[3, 1], perf[4, 1], perf[5, 1], perf[6, 1], perf[7, 1]
#             , perf[0, 2], perf[1, 2], perf[2, 2], perf[3, 2], perf[4, 2], perf[5, 2], perf[6, 2], perf[7, 2]
#             , perf[0, 3], perf[1, 3], perf[2, 3], perf[3, 3], perf[4, 3], perf[5, 3], perf[6, 3], perf[7, 3]
#             , perf[0, 4], perf[1, 4], perf[2, 4], perf[3, 4], perf[4, 4], perf[5, 4], perf[6, 4], perf[7, 4]],
#         'Legend': [str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5], str_1[6], str_1[7],
#                    str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5], str_1[6], str_1[7],
#                    str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5], str_1[6], str_1[7],
#                    str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5], str_1[6], str_1[7],
#                    str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5], str_1[6], str_1[7]]}
#     plt.figure()
#     sns.set_style("whitegrid")
#     sns.set(font_scale=0.8)
#     sns.barplot(x='No.of.records', y='Accuracy(%)', hue='Legend',palette=['#6600ff','#ff9900','#cc00ff','#99ff33','#1ab2ff','#ff6600','#669900','#cc3300'],data=df1)
#     plt.legend(ncol=2,loc='lower center', fontsize=15,prop={"weight":"bold"})
#     plt.xlabel(xlab, fontsize=15, weight='bold')
#     plt.ylabel(ylab, fontsize=15, weight='bold')
#     plt.savefig('Results_P1\\TP\\Comp_Analysis\\' + str(val) + '_' + str(name) +'Graph.png', dpi=800)
#     plt.show(block=False)
#     plt.clf()

def load_perf_value_saved_Algo_Analysis_1():
    perf_A = np.load('NPY\\COM_A.npy')
    perf_B = np.load('NPY\\COM_B.npy')
    perf_C = np.load('NPY\\COM_C.npy')
    perf_D = np.load('NPY\\COM_D.npy')
    perf_E = np.load('NPY\\COM_E.npy')
    perf_F = np.load('NPY\\COM_F.npy')
    perf_G = np.load('NPY\\COM_G.npy')
    perf_H = np.load('NPY\\COM_H.npy')
    perf_I = np.load('NPY\\COM_I.npy')

    A = np.asarray(perf_A[:][:])
    B = np.asarray(perf_B[:][:])
    C = np.asarray(perf_C[:][:])
    D = np.asarray(perf_D[:][:])
    E = np.asarray(perf_E[:][:])
    F = np.asarray(perf_F[:][:])
    G = np.asarray(perf_G[:][:])
    H = np.asarray(perf_H[:][:])
    I = np.asarray(perf_I[:][:])

    AA = A[:][:].transpose()
    BB = B[:][:].transpose()
    CC = C[:][:].transpose()
    DD = D[:][:].transpose()
    EE = E[:][:].transpose()
    FF = F[:][:].transpose()
    GG = G[:][:].transpose()
    HH = H[:][:].transpose()
    II = I[:][:].transpose()


    return [AA, BB, CC, DD, EE, FF, GG, HH, II]

def load_perf_value_saved_Algo_Analysis_2():
    perf_A = np.load('NPY1\\COM_A.npy')
    perf_B = np.load('NPY1\\COM_B.npy')
    perf_C = np.load('NPY1\\COM_C.npy')
    perf_D = np.load('NPY1\\COM_D.npy')
    perf_E = np.load('NPY1\\COM_E.npy')


    A = np.asarray(perf_A[:][:])
    B = np.asarray(perf_B[:][:])
    C = np.asarray(perf_C[:][:])
    D = np.asarray(perf_D[:][:])
    E = np.asarray(perf_E[:][:])


    AA = A[:][:].transpose()
    BB = B[:][:].transpose()
    CC = C[:][:].transpose()
    DD = D[:][:].transpose()
    EE = E[:][:].transpose()

    return [AA, BB, CC, DD, EE]

def Main_comp_val_acc_sen_spe_1(AA,BB,CC,DD,EE,FF,GG,HH,II):
    VALLL = np.column_stack((AA[0], BB[0], CC[0], DD[0], EE[0], FF[0], GG[0], HH[0], II[0]))
    perf1 = VALLL.T
    # perf1 = vall(perf1)
    VALLL = np.column_stack((AA[1], BB[1], CC[1], DD[1], EE[1], FF[1], GG[1], HH[1], II[1]))
    perf2 = VALLL.T
    # perf2 = vall(perf2)
    VALLL = np.column_stack((AA[2], BB[2], CC[2], DD[2], EE[2], FF[2], GG[2], HH[2], II[2]))
    perf3 = VALLL.T
    # perf3 = vall(perf3)
    VALLL = np.column_stack((AA[3], BB[3], CC[3], DD[3], EE[3], FF[3], GG[3], HH[3], II[3]))
    perf4 = VALLL.T
    # perf4 = vall(perf4)
    VALLL = np.column_stack((AA[4], BB[4], CC[4], DD[4], EE[4], FF[4], GG[4], HH[4], II[4]))
    perf5 = VALLL.T
    # perf5 = vall(perf5)
    VALLL = np.column_stack((AA[5], BB[5], CC[5], DD[5], EE[5], FF[5], GG[5], HH[5], II[5]))
    perf6 = VALLL.T
    VALLL = np.column_stack((AA[6], BB[6], CC[6], DD[6], EE[6], FF[6], GG[6], HH[6], II[6]))
    perf7 = VALLL.T
    VALLL = np.column_stack((AA[7], BB[7], CC[7], DD[7], EE[7], FF[7], GG[7], HH[7], II[7]))
    perf8 = VALLL.T
    VALLL = np.column_stack((AA[8], BB[8], CC[8], DD[8], EE[8], FF[8], GG[8], HH[8], II[8]))
    perf9 = VALLL.T
    # perf6 = vall(perf6)
    # [perf1, perf2, perf3, perf4, perf5, perf6, perf7, perf8, perf9,perf10,perf11,perf12]=perfs(perf1, perf2, perf3, perf4, perf5, perf6,perf7,perf8,perf9,perf10,perf11,perf12,0)
    return [perf1, perf2, perf3, perf4, perf5, perf6,perf7,perf8,perf9]

def Main_comp_val_acc_sen_spe_2(AA,BB,CC,DD,EE):
    VALLL = np.column_stack((AA[0], BB[0], CC[0], DD[0], EE[0]))
    perf1 = VALLL.T
    # perf1 = vall(perf1)
    VALLL = np.column_stack((AA[1], BB[1], CC[1], DD[1], EE[1]))
    perf2 = VALLL.T
    # perf2 = vall(perf2)
    VALLL = np.column_stack((AA[2], BB[2], CC[2], DD[2], EE[2]))
    perf3 = VALLL.T
    # perf3 = vall(perf3)
    VALLL = np.column_stack((AA[3], BB[3], CC[3], DD[3], EE[3]))
    perf4 = VALLL.T
    # perf4 = vall(perf4)
    VALLL = np.column_stack((AA[4], BB[4], CC[4], DD[4], EE[4]))
    perf5 = VALLL.T
    # perf5 = vall(perf5)
    VALLL = np.column_stack((AA[5], BB[5], CC[5], DD[5], EE[5]))
    perf6 = VALLL.T
    VALLL = np.column_stack((AA[6], BB[6], CC[6], DD[6], EE[6]))
    perf7 = VALLL.T
    VALLL = np.column_stack((AA[7], BB[7], CC[7], DD[7], EE[7]))
    perf8 = VALLL.T
    VALLL = np.column_stack((AA[8], BB[8], CC[8], DD[8], EE[8]))
    perf9 = VALLL.T
    # perf6 = vall(perf6)
    # [perf1, perf2, perf3, perf4, perf5, perf6, perf7, perf8, perf9,perf10,perf11,perf12]=perfs(perf1, perf2, perf3, perf4, perf5, perf6,perf7,perf8,perf9,perf10,perf11,perf12)

    return [perf1, perf2, perf3, perf4, perf5, perf6,perf7,perf8,perf9]

def load_perf_parameter1(A, B, C, D, E, F, G, H, I):
    perf_A1 = A[[0, 1, 2, 3, 4, 5, 6, 7, 8], :]
    perf_B1 = F[[0, 1, 2, 3, 4, 5, 6, 7, 8], :]
    perf_C1 = D[[0, 1, 2, 3, 4, 5, 6, 7, 8], :]
    perf_D1 = E[[0, 1, 2, 3, 4, 5, 6, 7, 8], :]
    perf_A2 = A[[8], :]  # Use index 8 instead of 9
    perf_B2 = F[[8], :]  # Use index 8 instead of 9
    perf_C2 = D[[8], :]  # Use index 8 instead of 9
    perf_D2 = E[[8], :]  # Use index 8 instead of 9
    return [perf_A1, perf_B1, perf_C1, perf_D1, perf_A2, perf_B2, perf_C2, perf_D2]



def load_perf_parameter2(A, B, C, D, E, F,G,H,I):
    perf_A1 = A
    perf_B1 = F
    perf_C1 = D
    perf_D1 = E
    return [perf_A1, perf_B1, perf_C1, perf_D1]

def Complete_Figure_perf(perf, val, str_1, xlab, ylab,name):
    perf = perf * 100
    a = perf[:, 0]
    b = perf[:, 1]
    c = perf[:, 2]
    d = perf[:, 3]
    e = perf[:, 4]
    dict = {'40': a, '50': b, '60': c, '70': d, '80': e}
    df = pd.DataFrame(dict, index=[str_1])
    df.to_csv('Results_P1\\TP\\Perf_Analysis\\_' + str(val) + '_' + str(name) +'Graph.csv')
    df1 = {
        'No.of.records': ['40', '40', '40', '40', '40',
                          '50', '50', '50', '50', '50',
                          '60', '60', '60', '60', '60',
                          '70', '70', '70', '70', '70',
                          '80', '80', '80', '80', '80'],
        'Accuracy(%)': [perf[0, 0], perf[1, 0], perf[2, 0], perf[3, 0], perf[4, 0]
            , perf[0, 1], perf[1, 1], perf[2, 1], perf[3, 1], perf[4, 1]
            , perf[0, 2], perf[1, 2], perf[2, 2], perf[3, 2], perf[4, 2]
            , perf[0, 3], perf[1, 3], perf[2, 3], perf[3, 3], perf[4, 3]
            , perf[0, 4], perf[1, 4], perf[2, 4], perf[3, 4], perf[4, 4]],
        'Legend': [str_1[0], str_1[1], str_1[2], str_1[3], str_1[4],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4]]}

    plt.figure()
    sns.set(font_scale=0.8)
    sns.set_style("whitegrid")
    sns.lineplot(x='No.of.records', y='Accuracy(%)', hue='Legend',  marker="d",markersize=10,palette=['#009900','#996600','#cc0000','#4da6ff','#6600ff'],data=df1)
    plt.legend(loc='lower center', fontsize=15,prop={'weight':'bold'})
    plt.xlabel(xlab, fontsize=15,weight='bold')
    plt.ylabel(ylab, fontsize=15,weight='bold')
    plt.savefig('Results_P1\\TP\\Perf_Analysis\\' + str(val) + '_' + str(name) +'Graph.png', dpi=800)
    plt.show(block=False)
    plt.clf()

def Complete_Figure_com_kf(perf, val, str_1, xlab, ylab,name):
    perf=perf*100
    a = perf[:, 0]
    b = perf[:, 1]
    c = perf[:, 2]
    d = perf[:, 3]
    dict = {'4': a, '6': b, '8': c, '10': d}
    df = pd.DataFrame(dict, index=[str_1])
    df.to_csv('Results_P1\\KF\\Comp_Analysis\\' + '_' + str(val) + str(name) +'_' + 'Graph.csv')
    df1 = {
        'No.of.records': ['4', '4', '4', '4', '4', '4', '4', '4',
                          '6', '6', '6', '6', '6', '6', '6', '6',
                          '8', '8', '8', '8', '8', '8', '8', '8',
                          '10', '10', '10', '10', '10', '10', '10', '10'],
        'Accuracy(%)': [perf[0, 0], perf[1, 0], perf[2, 0], perf[3, 0], perf[4, 0], perf[5, 0], perf[6, 0], perf[7, 0]
            , perf[0, 1], perf[1, 1], perf[2, 1], perf[3, 1], perf[4, 1], perf[5, 1], perf[6, 1], perf[7, 1]
            , perf[0, 2], perf[1, 2], perf[2, 2], perf[3, 2], perf[4, 2], perf[5, 2], perf[6, 2], perf[7, 2]
            , perf[0, 3], perf[1, 3], perf[2, 3], perf[3, 3], perf[4, 3], perf[5, 3], perf[6, 3], perf[7, 3]],
        'Legend': [str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5], str_1[6], str_1[7],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5], str_1[6], str_1[7],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5], str_1[6], str_1[7],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5], str_1[6], str_1[7],]}

    plt.figure()
    sns.set(font_scale=0.8)
    sns.set_style("whitegrid")
    sns.barplot(x='No.of.records', y='Accuracy(%)', hue='Legend', palette=['#6600ff','#ff9900','#cc00ff','#99ff33','#1ab2ff','#ff6600','#669900','#cc3300'],data=df1)
    plt.legend(ncol=2,loc='lower center', fontsize=15,prop={"weight":"bold"})
    plt.xlabel(xlab, fontsize=15, weight='bold')
    plt.ylabel(ylab, fontsize=15, weight='bold')
    plt.savefig('Results_P1\\KF\\Comp_Analysis\\' + str(val) + '_' + str(name) +'Graph.png', dpi=800)
    plt.show(block=False)
    plt.clf()

def complete_graph(ii):
    name = ["Accuracy", "F1-Score","Precision","Recall"]
    [AA, BB, CC, DD, EE, FF, GG, HH, II] = load_perf_value_saved_Algo_Analysis_1()
    [perf1, perf2, perf3, perf4, perf5, perf6,perf7,perf8,perf9] = Main_comp_val_acc_sen_spe_1(AA, BB, CC, DD, EE, FF, GG, HH, II)
    [perf_A1, perf_B1, perf_C1,perf_D1, perf_A2, perf_B2, perf_C2,perf_D2] = load_perf_parameter1(perf1, perf2, perf3, perf4, perf5,perf6,perf7,perf8,perf9)
    [AA, BB, CC, DD, EE] = load_perf_value_saved_Algo_Analysis_2()
    [perf1, perf2, perf3, perf4, perf5, perf6,perf7,perf8,perf9] = Main_comp_val_acc_sen_spe_2(AA, BB, CC, DD, EE)
    [perf_A,perf_B,perf_C,perf_D] = load_perf_parameter2(perf1, perf2, perf3, perf4,perf5,perf6,perf7,perf8,perf9)

    legend = ["DNN_Model", "Bi_LSTM_CNN ", "RNN ", "SVM_Model ", "Deep_CNN", "SGO-Deep_CNN"]
    legend1 = ["SGO-Deep_CNN with Epochs=20","SGO-Deep_CNN with Epochs=40","SGO-Deep_CNN with Epochs=60",
               "SGO-Deep_CNN with Epochs=80", "SGO-Deep_CNN with Epochs=100"]
    xlab = "Training Percentage(%)"
    ylab = name[0]+"(%)"
    Complete_Figure_com(perf_A1, ii, legend, xlab, ylab, name[0])
    ii = ii + 1
    ylab =  name[1]+"(%)"
    Complete_Figure_com(perf_B1, ii, legend, xlab, ylab, name[1])
    ii = ii + 1
    ylab = name[2]+"(%)"
    Complete_Figure_com(perf_C1, ii, legend, xlab, ylab, name[2])
    ii = ii + 1
    ylab =  name[3]+"(%)"
    Complete_Figure_com(perf_D1, ii, legend, xlab, ylab, name[3])
    ii = ii + 1
    ylab =   name[0]+"(%)"
    Complete_Figure_perf(perf_A2, ii, legend1, xlab, ylab, name[0])
    ii = ii + 1
    ylab =  name[1]+"(%)"
    Complete_Figure_perf(perf_B2, ii, legend1, xlab, ylab, name[1])
    ii = ii + 1
    ylab =  name[2]+"(%)"
    Complete_Figure_perf(perf_C2, ii, legend1, xlab, ylab, name[2])
    ii = ii + 1
    ylab =  name[3]+"(%)"
    Complete_Figure_perf(perf_D2, ii, legend1, xlab, ylab, name[3])
    ii = ii + 1

    xlab = "K-Fold"
    ylab =   name[0]+"(%)"
    Complete_Figure_com_kf(perf_A, ii, legend, xlab, ylab, name[0])
    ii = ii + 1
    ylab =  name[1]+"(%)"
    Complete_Figure_com_kf(perf_B, ii, legend, xlab, ylab, name[1])
    ii = ii + 1
    ylab =  name[2]+"(%)"
    Complete_Figure_com_kf(perf_C, ii, legend, xlab, ylab, name[2])
    ii = ii + 1
    ylab =  name[3]+"(%)"
    Complete_Figure_com_kf(perf_D, ii, legend, xlab, ylab, name[3])
def All_Analysis():
    # Features,labels = preprocess()
    print('\33[44m' + '\33[31m' + "[INFO] Loading Features and Labels" + '\x1b[0m')
    # Features = np.load("Datasets/features.npy")
    # labels = np.load("Datasets/labels.npy")
    # print('\33[46m' + '\33[30m' + "--------------[INFO] Training Percentage Analysis------------- " + '\x1b[0m')
    # TP_Analysis(Features,labels)
    # print('\33[46m' + '\33[30m' + "-------------[INFO] Cross Validation Analysis-------------- " + '\x1b[0m')
    # KF_Analysis(Features,labels)
    complete_graph(1)


All_Analysis()

