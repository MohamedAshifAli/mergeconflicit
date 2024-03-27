import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import BatchNormalization, Dropout, Flatten, Dense, Activation, MaxPooling2D, Conv2D, LSTM, \
    GlobalAveragePooling2D, MaxPool2D, multiply, Add, GlobalMaxPooling2D, Reshape,Lambda
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
# from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')
# data = pd.read_csv("Datasets/reduced_data_1.csv")
# data1=pd.read_csv("Datasets/reduced_data_2.csv")
# data2=pd.read_csv("Datasets/reduced_data_3.csv")
# data3=pd.read_csv("Datasets/reduced_data_4.csv")
# #
# concatenated_data = pd.concat([data, data1, data2, data3])
# concatenated_data.to_csv("concatenated_data.csv", index=False)

# Step 1: Load the CSV file into a DataFrame
concatenated_data = pd.read_csv("Datasets/concatenated_data.csv")[:1000]
concatenated_data.drop(columns=['pkSeqID'], inplace=True)
features = concatenated_data.drop(columns=['attack'])
labels = concatenated_data['attack']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Iterate over each column in features
for column in features.columns:
    # Check if the column contains non-numeric values
    if features[column].dtype == 'object':
        # Convert the column to numerical using LabelEncoder
        features[column] = label_encoder.fit_transform(features[column])

fet = features['flgs']
print(fet)


def split_data(ten_best_features,  target_features):
    train_size = 0.3
    total_data = len(ten_best_features)
    indices = np.arange(total_data)
    np.random.shuffle(indices)

    split_index = int(total_data * (1 - train_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    xtrain = ten_best_features[train_indices]
    xtest = ten_best_features[test_indices]
    ytrain = target_features[train_indices]
    ytest = target_features[test_indices]

    return xtrain, xtest, ytrain, ytest

# Define your CNN model
def get_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(CONST.IMG_SIZE, CONST.IMG_SIZE, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Model prepared...')
    return model


# Create an instance of your CNN model
model = get_model()

