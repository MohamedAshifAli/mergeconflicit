import os

from imblearn.over_sampling import RandomOverSampler
from keras import Sequential, Model, Input
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, MaxPooling2D, Concatenate, Conv1D, Reshape, LSTM, \
    Bidirectional
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import seaborn as sns
from tensorflow import keras
from termcolor import colored


def coloumn_values_to_integer(features):
    label_encoder = LabelEncoder()
    for column in features.columns:
        if features[column].dtype == 'object':
            try:
                features[column] = label_encoder.fit_transform(features[column])
            except:
                features[column] = label_encoder.fit_transform(features[column].astype("str"))
    return features

def outlier(features):
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(features)
    unique_values, counts = np.unique(yhat, return_counts=True)
    outlier_ind = np.max(yhat)
    features['Outlier_Label'] = yhat
    features = features[yhat == outlier_ind]
    return features

#Traffic data
def db1_preprocess():
    db1 = pd.read_csv("Datasets/traffic.csv")
    db1.head()
    db1['DateTime']=pd.to_datetime(db1['DateTime'])
    db1["Year"]=db1['DateTime'].dt.year
    db1["Month"]=db1['DateTime'].dt.month
    db1["Date_no"]=db1['DateTime'].dt.day
    db1["Hour"]=db1['DateTime'].dt.hour
    db1["Day"]= db1['DateTime'].dt.strftime("%A")
    db1.drop("DateTime", axis=1, inplace=True)
    db1.dropna()
    db1.drop_duplicates()
    features = coloumn_values_to_integer(db1)
    features =outlier(features)
    features = features.drop(columns=['Outlier_Label'])
    lab = LabelEncoder()
    features['Vehicles'].value_counts()
    features['Vehicles'] = lab.fit_transform(features['Vehicles'])  # convert text to numbers
    features['Vehicles'].value_counts()
    labels = features['Vehicles']
    features = features.astype('float32') / features.max()
    features = np.array(features)
    labels = np.array(labels)
    np.save('db1_features', features)
    np.save('db1_labels', labels)
    return features,labels

#Air
def db2_preprocess():
    db2 = pd.read_csv("Datasets/AirQuality.csv", sep=";", decimal=",")
    db2 = db2.loc[:, ~db2.columns.str.contains('^Unnamed')]
    db2.replace(-200, np.nan, inplace=True)
    db2=db2.fillna(0)
    coloumn_values_to_integer(db2)
    outlier(db2)
    features = db2.drop(columns=['Outlier_Label', 'Date', 'Time'])
    labels = features[['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']]
    lab = LabelEncoder()
    for col in labels.columns:
        labels[col] = lab.fit_transform(labels[col])
    labels = labels[['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']].mean(axis=1)
    features = features.astype('float32') / features.max()
    features = np.array(features)
    labels = np.array(labels)
    np.save('db2_features', features)
    np.save('db2_labels', labels)
    return features,labels



def db3_preprocess():
    db3 = pd.read_csv("Datasets/Life_Expectancy_Data.csv")
    db3 = db3.fillna(0)
    coloumn_values_to_integer(db3)
    outlier(db3)
    features = db3.drop(columns=['Outlier_Label'])
    lab = LabelEncoder()
    features['Life expectancy ']= lab.fit_transform(features['Life expectancy '])  # convert text to numbers
    labels = features['Life expectancy ']
    features = features.astype('float32') / features.max()
    features = np.array(features)
    labels = np.array(labels)
    np.save('db3_features', features)
    np.save('db3_labels', labels)
    return features,labels


def splits(features, labels,  train_size):
    xtrain1,xtest1,ytrain1,ytest1=[],[],[],[]
    for i in range(len(np.unique(labels))):
        indices = np.where(labels == i)[0]
        feat=features[indices]
        lab=labels[indices]
        total_data = len(feat)
        indices = np.arange(total_data)
        np.random.shuffle(indices)
        split_index = int(total_data * (train_size))
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]
        xtrain1.append(feat[train_indices])
        ytrain1.append(lab[train_indices])
        xtest1.append(feat[test_indices])
        ytest1.append(lab[test_indices])
    xtrain=np.vstack(xtrain1)
    xtest=np.vstack(xtest1)
    ytrain=np.hstack(ytrain1)
    ytest=np.hstack(ytest1)
    return xtrain, ytrain, xtest, ytest


#Compartve
def K_Nearest_Classifier(xtrain, ytrain, xtest, ytest):
    Model_name = "K_Nearest_Classifier"
    print('\033[46m' + '\033[30m' + "------" + Model_name + "-----" + '\x1b[0m')
    knn_model = KNeighborsRegressor().fit(xtrain, ytrain)
    Y_pred = knn_model.predict(xtest)
    return ytest,Y_pred

def DNN(xtrain, xtest, ytrain, ytest,epochs):
    Model_name = "DNN"
    print('\033[46m' + '\033[30m' + "------" + Model_name + "-----" + '\x1b[0m')
    model = Sequential()
    model.add(Dense(512))
    model.add(Dense(512))
    model.add(Dense(256))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(xtrain, ytrain, batch_size=32, epochs=epochs, verbose=1)
    Y_pred = model.predict(xtest)
    Y_pred = model.predict(xtest)
    Y_pred = Y_pred.flatten()
    return Y_pred,ytest
#
# def CNN_LSTM(xtrain, xtest, ytrain, ytest, epochs):
#     Model_name = "CNN_LSTM"
#     print('\033[46m' + '\033[30m' + "------" + Model_name + "-----" + '\x1b[0m')
#     xtrain = xtrain.astype('float32') / xtrain.max()
#     xtest = xtest.astype('float32') / xtest.max()
#     xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1, 1)
#     xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1, 1)
#     input_layer = Input(shape=(xtrain.shape[1], xtrain.shape[2], xtrain.shape[3]))
#     x = Conv2D(8, (1, 1), activation='relu')(input_layer)
#     x = MaxPooling2D((1, 1))(x)
#     x = Conv2D(8, (1, 1), activation='relu')(x)
#     x = MaxPooling2D((1, 1))(x)
#     x = Conv2D(16, (1, 1), activation='relu')(x)
#     x = MaxPooling2D((1, 1))(x)
#     x = Conv2D(32, (1, 1), activation='relu')(x)
#     x = MaxPooling2D((1, 1))(x)
#     x = Flatten()(x)
#     x = Reshape((x.shape[1], 1))(x)
#     x = LSTM(32, return_sequences=True, activation='relu')(x)
#     x = Flatten()(x)
#     x = Dense(128, activation='relu')(x)
#     output_layer = Dense(1, activation='linear')(x)
#     model = Model(inputs=input_layer, outputs=output_layer)
#     # optimizer = Adam(learning_rate=0.001)
#     optimizer = Adam
#     model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
#     model.fit(xtrain, ytrain, batch_size=128, epochs=epochs, verbose=1)
#     Y_pred = model.predict(xtest)
#     return Y_pred, ytest

def CNN_LSTM(xtrain, xtest, ytrain, ytest, epochs):
    Model_name = "CNN_LSTM"
    print('\033[46m' + '\033[30m' + "------" + Model_name + "-----" + '\x1b[0m')
    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1, 1)
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1, 1)
    xtrain = xtrain.astype('float32') / xtrain.max()
    xtest = xtest.astype('float32') / xtest.max()

    model = Sequential()
    model.add(Conv2D(8, (1, 1), activation='relu', input_shape=(xtrain.shape[1], xtrain.shape[2], 1)))
    model.add(MaxPooling2D((1, 1)))
    model.add(Conv2D(8, (1, 1), activation='relu'))
    model.add(MaxPooling2D((1, 1)))
    model.add(Conv2D(16, (1, 1), activation='relu'))
    model.add(MaxPooling2D((1, 1)))
    model.add(Conv2D(32, (1, 1), activation='relu'))
    model.add(MaxPooling2D((1, 1)))
    model.add(Flatten())
    model.add(Reshape((model.output_shape[1], 1)))
    model.add(LSTM(32, return_sequences=True, activation='tanh'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    # Reduce learning rate when a metric has stopped improving
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    history = model.fit(xtrain, ytrain, batch_size=64, epochs=epochs, verbose=1
                        )
    Y_pred = model.predict(xtest)
    Y_pred = Y_pred.flatten()
    return Y_pred, ytest

def BiLSTM(xtrain, xtest, ytrain, ytest,epochs):
    Model_name = "BiLSTM"
    print('\033[46m' + '\033[30m' + "------" + Model_name + "-----" + '\x1b[0m')

    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)

    input_layer = Input(shape=(xtrain.shape[1],xtrain.shape[2]))
    x = Bidirectional(LSTM(100))(input_layer)
    x = Dense(16, activation='linear')(x)
    output_layer = Dense(1, activation='linear')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(xtrain, ytrain, batch_size=32, epochs=epochs, verbose=1)
    Y_pred = model.predict(xtest)
    Y_pred = Y_pred.flatten()
    return Y_pred,ytest


#PROPOSED
def Meta_Learning_And_Adaptive_Learning(xtrain, ytrain, xtest, ytest,epochs):
    # xtrain = xtrain.astype('float32') / xtrain.max()
    # xtest = xtest.astype('float32') / xtest.max()
    # ytrain = ytrain.astype("float32")/ytrain.max()
    # ytest = ytest.astype("float32")/ytest.max()
    # xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1, 1)
    # xtest = xtrain.reshape(xtest.shape[0], xtest.shape[1], 1, 1)

    xtrain = np.expand_dims(xtrain, axis=2)
    xtrain = np.expand_dims(xtrain, axis=3)
    xtest = np.expand_dims(xtest, axis=2)
    xtest = np.expand_dims(xtest, axis=3)
    def meta_model():
        print(colored("Convolutional Neural Network>> ", color='blue', on_color='on_grey'))
        model = keras.Sequential()
        model.add(
            Conv2D(32, (1, 1), activation='relu', input_shape=(xtrain.shape[1], xtrain.shape[2], xtrain.shape[3])))
        model.add(MaxPooling2D((1, 1)))
        model.add(Conv2D(32, (1, 1), activation='relu'))
        model.add(MaxPooling2D((1, 1)))
        model.add(Conv2D(64, (1, 1), activation='relu'))
        model.add(MaxPooling2D((1, 1)))
        # model.add(Conv2D(16, (1, 1), activation='relu'))
        # model.add(MaxPooling2D((1, 1)))
        # model.add(Conv2D(32, (1, 1), activation='relu'))
        # model.add(MaxPooling2D((1, 1)))
        model.add(Flatten())
        # model.add(Dense(2048, activation='linear'))
        # model.add(Dense(1024, activation='linear'))
        # model.add(Dense(512, activation='linear'))
        model.add(Dense(128, activation='linear'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    # Meta-learning parameters
    num_tasks = 3  # Number of meta-tasks
    num_epochs_inner = 5  # Number of epochs for inner training
    # Split your training data into chunks of tasks
    all_data = xtrain.shape[0] // num_tasks
    x1train = [xtrain[i * all_data:(i + 1) * all_data] for i in range(num_tasks)]
    y1train = [ytrain[i * all_data:(i + 1) * all_data] for i in range(num_tasks)]
    meta_model_1 = meta_model()

    # Meta-training loop
    for task in range(num_tasks):
        print('\033[46m' + '\033[30m' + "________________________.Meta-training loop.__________________________________" + '\x1b[0m'+"-"+str(task))
        shuffled_indices = np.random.permutation(num_tasks)
        task_X_train = x1train[task ]
        task_y_train = y1train[task ]
        task_X_train = np.expand_dims(task_X_train, axis=-1)
        # task_y_train = np.expand_dims(task_y_train, axis=-1)
        # Inner loop (training on task-specific data)
        for epoch in range(num_epochs_inner):
            print('\033[46m' + '\033[30m' + "________________________.Inner loop.__________________________________" + '\x1b[0m' + "-" + str(epoch))

            history = meta_model_1.fit(task_X_train, task_y_train, epochs=epochs, batch_size=32, verbose=1)
            accuracy_mean = history.history['mae']
            accuracy_mean = np.mean(accuracy_mean)
            print(accuracy_mean)

            threshold_value = 10
            while accuracy_mean >= threshold_value:
                print(colored(f"Model Training Epoch {epoch + 1} >> ", color='blue', on_color='on_grey'))
                history = meta_model_1.fit(task_X_train, task_y_train, epochs=epochs, batch_size=128,verbose =1)
                accuracy_mean = history.history['mae']
                accuracy_mean = np.mean(accuracy_mean)
                print('\033[46m' + '\033[30m' + "________________________Training accurcy value__________________________________" + '\x1b[0m' + "-" + str(accuracy_mean))

        print(f"Task {task + 1}/{num_tasks} completed.")
    print("Meta-training finished.")
    # Now, make predictions on the test data
    ypred = meta_model_1.predict(xtest)
    ypred_flat = ypred.flatten()


    return ypred_flat,ytest

def prepprocessing():
    db1_preprocess()
    db2_preprocess()
    db3_preprocess()

def parameter(mae, mse, rmse):
    mae = np.mean(mae)
    mse = np.mean(mse)
    rmse = np.mean(rmse)
    return [mae, mse, rmse]

def calculate_metrics(pred,true):
    mae = mean_absolute_error(true,pred)
    mse = mean_squared_error(true,pred)
    rmse = np.sqrt(mse)
    return mae,mse,rmse


def TP_Analysis():

    features = np.load('feat and lab/db1_features.npy')
    labels = np.load('feat and lab/db1_labels.npy')
    # ros = RandomOverSampler(random_state=42)
    # # Reshape oversampled features
    # X, y = ros.fit_resample(xtrain, y_train)
    tr = [0.4,0.5,0.6,0.7,0.8]
    # epochs = [20,40,60,80,100]
    epochs = [4,4,4,4,4]
    # features = features.astype("float32")/features.max()
    # labels = labels.astype("float32")/labels.max()
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

        xtrain, ytrain, xtest, ytest = splits(features, labels, train_size=tr[p])

        Y_pred1, Y_true1 = K_Nearest_Classifier(xtrain, ytrain, xtest, ytest)
        Y_pred2, Y_true2 = DNN(xtrain, xtest, ytrain, ytest, epochs[4])
        Y_pred3, Y_true3 = BiLSTM(xtrain, xtest, ytrain, ytest, epochs[4])
        Y_pred4, Y_true4= CNN_LSTM(xtrain, xtest, ytrain, ytest, epochs[0])

        Y_pred5, Y_true5 = Meta_Learning_And_Adaptive_Learning(xtrain, ytrain, xtest, ytest, 4)
        Y_pred6, Y_true6 = Meta_Learning_And_Adaptive_Learning(xtrain, ytrain, xtest, ytest, 8)
        Y_pred7, Y_true7 = Meta_Learning_And_Adaptive_Learning(xtrain, ytrain, xtest, ytest, 12)
        Y_pred8, Y_true8 = Meta_Learning_And_Adaptive_Learning(xtrain, ytrain, xtest, ytest,16)
        Y_pred9, Y_true9 = Meta_Learning_And_Adaptive_Learning(xtrain, ytrain, xtest, ytest, 20)



        [MAE1, MSE1, RMSE1] = calculate_metrics(Y_pred1, Y_true1)
        [MAE2, MSE2, RMSE2] = calculate_metrics(Y_pred2, Y_true2)
        [MAE3, MSE3, RMSE3] = calculate_metrics(Y_pred3, Y_true3)
        [MAE4, MSE4, RMSE4] = calculate_metrics(Y_pred4, Y_true4)
        [MAE5, MSE5, RMSE5] = calculate_metrics(Y_pred5, Y_true5)
        [MAE6, MSE6, RMSE6] = calculate_metrics(Y_pred6, Y_true6)
        [MAE7, MSE7, RMSE7] = calculate_metrics(Y_pred7, Y_true7)
        [MAE8, MSE8, RMSE8] = calculate_metrics(Y_pred8, Y_true8)
        [MAE9, MSE9, RMSE9] = calculate_metrics(Y_pred9, Y_true9)

        COM_A.append([MAE1, MSE1, RMSE1])
        COM_A.append([MAE2, MSE2, RMSE2])
        COM_A.append([MAE3, MSE3, RMSE3])
        COM_A.append([MAE4, MSE4, RMSE4])
        COM_A.append([MAE5, MSE5, RMSE5])
        COM_A.append([MAE6, MSE6, RMSE6])
        COM_A.append([MAE7, MSE7, RMSE7])
        COM_A.append([MAE8, MSE8, RMSE8])
        COM_A.append([MAE9, MSE9, RMSE9])

    np.save('NPY\\COM_A.npy'.format(os.getcwd()),COM_A)
    np.save('NPY\\COM_B.npy'.format(os.getcwd()),COM_B)
    np.save('NPY\\COM_C.npy'.format(os.getcwd()),COM_C)
    np.save('NPY\\COM_D.npy'.format(os.getcwd()),COM_D)
    np.save('NPY\\COM_E.npy'.format(os.getcwd()),COM_E)
    np.save('NPY\\COM_F.npy'.format(os.getcwd()),COM_F)
    np.save('NPY\\COM_G.npy'.format(os.getcwd()),COM_G)
    np.save('NPY\\COM_H.npy'.format(os.getcwd()),COM_H)
    np.save('NPY\\COM_I.npy'.format(os.getcwd()),COM_I)


def  KF_Analysis(features,labels):

    # features, labels = oversample(features, labels)
    """K-fold cross-validation approach divides the input dataset into K groups of samples of equal sizes.
     These samples are called folds. For each learning set, the prediction function uses k-Cotton Leaf folds,
      and the rest of the folds are used for the test set. This approach is a very popular CV approach
      because it is easy to understand, and the output is less biased than other methods."""
    kr = [4, 6, 8, 10]
    """An epoch in machine learning means one complete pass of the training dataset through the algorithm.
         This epoch's number is an important hyperparameter for the algorithm. 
         It specifies the number of epochs or complete passes of the entire training dataset passing through the training or learning process of the algorithm"""

    epochs = [3,3,3,3,3]

    """A label represents an output value,
       while a feature is an input value that describes the characteristics of such labels in datasets"""
    features = np.nan_to_num(features, 0)

    """Normalization is a technique often applied as part of data preparation for machine learning. 
        The goal of normalization is to change the values of numeric columns in the dataset to use a common scale,
         without distorting differences in the ranges of values or losing information"""
    ## Normalization
    # feat = features.astype(np.float32) / features.max()

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
        kfold = strtfdKFold.split(features, labels)

        mae1, mse1, rmse1 = [], [], []
        mae2, mse2, rmse2 = [], [], []
        mae3, mse3, rmse3 = [], [], []
        mae4, mse4, rmse4 = [], [], []
        mae5, mse5, rmse5 = [], [], []

        for k, (train, test) in enumerate(kfold):
            tr_data = features[train, :]
            ytrain = labels[train]
            tst_data = features[test, :]
            ytest = labels[test]
            xtrain, xtest = tr_data, tst_data

            # print('\033[46m' + '\033[30m' + "------------" + "MODEL TRAINING SECTION" + "------------" + '\x1b[0m')

        Y_pred1, Y_true1 = K_Nearest_Classifier(xtrain, ytrain, xtest, ytest)
        Y_pred2, Y_true2 = DNN(xtrain, xtest, ytrain, ytest, epochs[4])
        Y_pred3, Y_true3 = BiLSTM(xtrain, xtest, ytrain, ytest, epochs[4])
        Y_pred4, Y_true4 = CNN_LSTM(xtrain, xtest, ytrain, ytest, epochs[4])
        Y_pred5, Y_true5 = Meta_Learning_And_Adaptive_Learning(xtrain, ytrain, xtest, ytest, 10)


        print('\033[46m' + '\033[30m' + "------------" + "Metrics Evaluated from Confusion Matrix" + "------------" + '\x1b[0m')

        [MAE1, MSE1, RMSE1] = calculate_metrics(Y_pred1, Y_true1)
        [MAE2, MSE2, RMSE2] = calculate_metrics(Y_pred2, Y_true2)
        [MAE3, MSE3, RMSE3] = calculate_metrics(Y_pred3, Y_true3)
        [MAE4, MSE4, RMSE4] = calculate_metrics(Y_pred4, Y_true4)
        [MAE5, MSE5, RMSE5] = calculate_metrics(Y_pred5, Y_true5)

        mae1.append(MAE1)
        mse1.append(MSE1)
        rmse1.append(RMSE1)
        mae2.append(MAE2)
        mse2.append(MSE2)
        rmse2.append(RMSE2)
        mae3.append(MAE3)
        mse3.append(MSE3)
        rmse3.append(RMSE3)
        mae4.append(MAE4)
        mse4.append(MSE4)
        rmse4.append(RMSE4)
        mae5.append(MAE5)
        mse5.append(MSE5)
        rmse5.append(RMSE5)

    [MAE1, MSE1, RMSE1] =parameter(mae1, mse1, rmse1)
    [MAE2, MSE2, RMSE2] =parameter(mae2, mse2, rmse2)
    [MAE3, MSE3, RMSE3] = parameter(mae3, mse3, rmse3)
    [MAE4, MSE4, RMSE4] = parameter(mae4, mse4, rmse4)
    [MAE5, MSE5, RMSE5] = parameter(mae5, mse5, rmse5)


    COM_A.append([MAE1, MSE1, RMSE1])
    COM_B.append( [MAE2, MSE2, RMSE2])
    COM_C.append([MAE3, MSE3, RMSE3])
    COM_D.append([MAE4, MSE4, RMSE4])
    COM_E.append([MAE5, MSE5, RMSE5])

    np.save('NPY1\\COM_A.npy'.format(os.getcwd()),COM_A)
    np.save('NPY1\\COM_B.npy'.format(os.getcwd()),COM_B)
    np.save('NPY1\\COM_C.npy'.format(os.getcwd()),COM_C)
    np.save('NPY1\\COM_D.npy'.format(os.getcwd()),COM_D)
    np.save('NPY1\\COM_E.npy'.format(os.getcwd()),COM_E)

features = np.load('feat and lab/db1_features.npy')
labels = np.load('feat and lab/db1_labels.npy')
TP_Analysis()
KF_Analysis(features,labels)
print()

def Complete_Figure_com(perf, val, str_1, xlab, ylab,name):
    perf = perf * 100
    a = perf[:, 0]
    b = perf[:, 1]
    c = perf[:, 2]
    d = perf[:, 3]
    e= perf[:,4]
    dict = {'40': a, '50': b, '60': c, '70': d,'80':e}
    df = pd.DataFrame(dict, index=[str_1])
    df.to_csv('Results_P1\\TP\\Comp_Analysis\\' + '_' + str(val) + str(name) +'_' + 'Graph.csv')
    df1 = {
        'No.of.records': ['40', '40', '40', '40', '40', '40',
                          '50', '50', '50', '50', '50', '50',
                          '60', '60', '60', '60', '60', '60',
                          '70', '70', '70', '70', '70', '70',
                          '80', '80', '80', '80', '80', '80'],
        'Accuracy(%)': [perf[0, 0], perf[1, 0], perf[2, 0], perf[3, 0], perf[4, 0], perf[5, 0]
            , perf[0, 1], perf[1, 1], perf[2, 1], perf[3, 1], perf[4, 1], perf[5, 1]
            , perf[0, 2], perf[1, 2], perf[2, 2], perf[3, 2], perf[4, 2], perf[5, 2]
            , perf[0, 3], perf[1, 3], perf[2, 3], perf[3, 3], perf[4, 3], perf[5, 3]
            , perf[0, 4], perf[1, 4], perf[2, 4], perf[3, 4], perf[4, 4], perf[5, 4]],
        'Legend': [str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5]]}
    plt.figure()
    sns.set_style("whitegrid")
    sns.set(font_scale=0.8)
    sns.barplot(x='No.of.records', y='Accuracy(%)', hue='Legend',palette=['#6600ff','#ff9900','#cc00ff','#99ff33','#1ab2ff','#ff6600'],data=df1)
    plt.legend(ncol=2,loc='lower center', fontsize=18,prop={"weight":"bold"})
    plt.xlabel(xlab, fontsize=15, weight='bold')
    plt.ylabel(ylab, fontsize=15, weight='bold')
    plt.savefig('Results_P1\\TP\\Comp_Analysis\\' + str(val) + '_' + str(name) +'Graph.png', dpi=800)
    plt.show(block=False)
    plt.clf()


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
    VALLL = np.column_stack((AA[1], BB[1], CC[1], DD[1], EE[1], FF[1], GG[1], HH[1], II[1]))
    perf2 = VALLL.T
    VALLL = np.column_stack((AA[2], BB[2], CC[2], DD[2], EE[2], FF[2], GG[2], HH[2], II[2]))
    perf3 = VALLL.T
    VALLL = np.column_stack((AA[3], BB[3], CC[3], DD[3], EE[3], FF[3], GG[3], HH[3], II[3]))
    perf4 = VALLL.T
    VALLL = np.column_stack((AA[4], BB[4], CC[4], DD[4], EE[4], FF[4], GG[4], HH[4], II[4]))
    perf5 = VALLL.T
    VALLL = np.column_stack((AA[5], BB[5], CC[5], DD[5], EE[5], FF[5], GG[5], HH[5], II[5]))
    perf6 = VALLL.T
    VALLL = np.column_stack((AA[6], BB[6], CC[6], DD[6], EE[6], FF[6], GG[6], HH[6], II[6]))
    perf7 = VALLL.T
    VALLL = np.column_stack((AA[7], BB[7], CC[7], DD[7], EE[7], FF[7], GG[7], HH[7], II[7]))
    perf8 = VALLL.T
    VALLL = np.column_stack((AA[8], BB[8], CC[8], DD[8], EE[8], FF[8], GG[8], HH[8], II[8]))
    perf9 = VALLL.T
    VALLL = np.column_stack((AA[9], BB[9], CC[9], DD[9], EE[9], FF[9], GG[9], HH[9], II[9]))
    perf10 = VALLL.T
    VALLL = np.column_stack((AA[10], BB[10], CC[10], DD[10], EE[10], FF[10], GG[10], HH[10], II[10]))
    perf11 = VALLL.T
    VALLL = np.column_stack((AA[11], BB[11], CC[11], DD[11], EE[11], FF[11], GG[11], HH[11], II[11]))
    perf12 = VALLL.T
    # [perf1, perf2, perf3, perf4, perf5, perf6,perf7,perf8,perf9,perf10,perf11,perf12]=perfs(perf1, perf2, perf3, perf4, perf5, perf6,perf7,perf8,perf9,perf10,perf11,perf12,1)
    return [perf1, perf2, perf3, perf4, perf5, perf6,perf7,perf8,perf9,perf10,perf11,perf12]



def Main_comp_val_acc_sen_spe_2(AA,BB,CC,DD,EE):
    VALLL = np.column_stack((AA[0], BB[0], CC[0], DD[0], EE[0]))
    perf1 = VALLL.T
    VALLL = np.column_stack((AA[1], BB[1], CC[1], DD[1], EE[1]))
    perf2 = VALLL.T
    VALLL = np.column_stack((AA[2], BB[2], CC[2], DD[2], EE[2]))
    perf3 = VALLL.T
    VALLL = np.column_stack((AA[3], BB[3], CC[3], DD[3], EE[3]))
    perf4 = VALLL.T
    VALLL = np.column_stack((AA[4], BB[4], CC[4], DD[4], EE[4]))
    perf5 = VALLL.T
    VALLL = np.column_stack((AA[5], BB[5], CC[5], DD[5], EE[5]))
    perf6 = VALLL.T
    VALLL = np.column_stack((AA[6], BB[6], CC[6], DD[6], EE[6]))
    perf7 = VALLL.T
    VALLL = np.column_stack((AA[7], BB[7], CC[7], DD[7], EE[7]))
    perf8 = VALLL.T
    VALLL = np.column_stack((AA[8], BB[8], CC[8], DD[8], EE[8]))
    perf9 = VALLL.T
    VALLL = np.column_stack((AA[9], BB[9], CC[9], DD[9], EE[9]))
    perf10 = VALLL.T
    VALLL = np.column_stack((AA[10], BB[10], CC[10], DD[10], EE[10]))
    perf11 = VALLL.T
    VALLL = np.column_stack((AA[11], BB[11], CC[11], DD[11], EE[11]))
    perf12 = VALLL.T
    # [perf1, perf2, perf3, perf4, perf5, perf6,perf7,perf8,perf9,perf10,perf11,perf12]=perfs(perf1, perf2, perf3, perf4, perf5, perf6,perf7,perf8,perf9,perf10,perf11,perf12,2)
    return [perf1, perf2, perf3, perf4, perf5, perf6,perf7,perf8,perf9,perf10,perf11,perf12]

def load_perf_parameter1(A, B, C, D, E, F, G, H, I):
    perf_A1 = A[[0, 1, 2, 3, 8], :]
    perf_B1 = F[[0, 1, 2, 3, 8], :]
    perf_C1 = D[[0, 1, 2, 3, 8], :]
    perf_D1 = E[[0, 1, 2, 3, 8], :]
    perf_A2 = A[[4, 5, 6, 7, 8], :]  # Use index 8 instead of 9
    perf_B2 = F[[4, 5, 6, 7, 8], :]  # Use index 8 instead of 9
    perf_C2 = D[[4, 5, 6, 7, 8], :]  # Use index 8 instead of 9
    perf_D2 = E[[4, 5, 6, 7, 8], :]  # Use index 8 instead of 9
    return [perf_A1, perf_B1, perf_C1, perf_D1, perf_A2, perf_B2, perf_C2, perf_D2]

def load_perf_parameter2(A, B, C, D, E, F,G,H,I):
    perf_A1 = A
    perf_B1 = F
    perf_C1 = D
    perf_D1 = E
    return [perf_A1, perf_B1, perf_C1, perf_D1]

# db1_preprocess()
# db2_preprocess()
# db3_preprocess()

