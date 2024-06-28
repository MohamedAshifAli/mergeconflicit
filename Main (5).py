import os

import ReliefF
import TSFEDL
import librosa
import numpy as np
import matplotlib.pyplot as plt
from fiftyone import Flatten
from keras import Sequential, Input, Model
from keras.layers import Conv1D, Dense, MaxPooling1D, Conv2D, MaxPooling2D, LeakyReLU, Dropout, Reshape, LSTM, \
    Concatenate
from keras.utils import to_categorical, plot_model

from scipy.signal import butter, filtfilt
import scipy.io
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
from termcolor import colored

from mealpy import FloatVar

# Function for Butterworth bandpass filter
directory = glob("trainingnew\\training\\*.mat")
# from scipy.signal import find_peaks

from scipy.signal import butter, filtfilt
import scipy.io
from glob import glob

# Function for Butterworth bandpass filter


# Derivative filter
def derivative_filter(data):
    derivative_pass = np.diff(data, prepend=data[0])
    return derivative_pass

# Squaring function
def squaring(data):
    square_pass = data ** 2
    return square_pass

# Moving average filter
def moving_window_integration(data, sample_rate, window_size=None):
    if window_size is None:
        assert sample_rate is not None, "if window size is None, sampling rate should be given"
        window_size = int(0.08 * int(sample_rate))
    integrated_signal = np.zeros(len(data))
    cumulative_sum = np.cumsum(data)
    integrated_signal[window_size:] = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    integrated_signal[:window_size] = cumulative_sum[:window_size] / np.arange(1, window_size + 1)
    return integrated_signal
from scipy.signal import butter, filtfilt

def main_est_perf_metrics(preds, y_test):
    """
    Calculate performance metrics based on the predictions and true labels.

    Args:
        preds: Predicted labels.
        y_test: True labels.

    Returns:
        List: List of performance metrics including accuracy, sensitivity, specificity, precision,
              recall, F1 score, Critical Success Index (CSI), False Positive Rate (FPR),
              False Negative Rate (FNR), Matthews Correlation Coefficient (MCC), Negative Predictive Value (NPV),
              and Positive Predictive Value (PPV).
    """


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
# Define Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def differentiate(signal):
    return np.diff(signal)

def find_pqrst_waves(directory):
    fs = 1000

def calculate_threshold(signal, threshold_factor=0.5):
        x = np.mean(signal) * threshold_factor
        return x

# def find_peaks(diff_signal, threshold):
#     peaks, _ = scipy.signal.find_peaks(diff_signal[0], height=threshold)
#     return peaks

def find_peaks(diff_signal, threshold):
    all_peaks = []
    for i in range(diff_signal.shape[0]):
        peaks, _ = scipy.signal.find_peaks(diff_signal[i], height=threshold)
        all_peaks.append(peaks)
    return all_peaks

def mark_pqrst(filtered_signal, peaks):
    p_waves = []
    q_waves = []
    r_waves = []
    s_waves = []
    t_waves = []

    for row, peak_list in enumerate(peaks):
        row_p_waves = []
        row_q_waves = []
        row_r_waves = []
        row_s_waves = []
        row_t_waves = []

        if len(peak_list) == 0:
            # Append empty lists if no peaks are detected in this row
            p_waves.append(row_p_waves)
            q_waves.append(row_q_waves)
            r_waves.append(row_r_waves)
            s_waves.append(row_s_waves)
            t_waves.append(row_t_waves)
            continue  # Move to the next row

        r_wave = peak_list[0]  # Initialize with the first peak

        for i in range(1, len(peak_list)):
            if filtered_signal[row][peak_list[i]] > filtered_signal[row][r_wave]:
                r_wave = peak_list[i]

        row_r_waves.append(r_wave)

        q_candidates = [p for p in peak_list if p < r_wave and filtered_signal[row][p] < 0]
        if q_candidates:
            row_q_waves.append(q_candidates[-1])

        s_candidates = [p for p in peak_list if p > r_wave and filtered_signal[row][p] < 0]
        if s_candidates:
            row_s_waves.append(s_candidates[0])

        if row_q_waves:
            p_candidates = [p for p in peak_list if p < row_q_waves[-1]]
            if p_candidates:
                row_p_waves.append(p_candidates[-1])

        if row_s_waves:
            t_candidates = [p for p in peak_list if p > row_s_waves[-1]]
            if t_candidates:
                row_t_waves.append(t_candidates[0])

        # Append detected waves for this row
        p_waves.append(row_p_waves)
        q_waves.append(row_q_waves)
        r_waves.append(row_r_waves)
        s_waves.append(row_s_waves)
        t_waves.append(row_t_waves)

    return p_waves, q_waves, r_waves, s_waves, t_waves

def detect_conditions_ppg(signal, fs):

    conditions = []

    # Check for Asystole: No QRS for at least 4 seconds
    qrs_intervals = find_qrs_intervals(signal, fs)
    if len(qrs_intervals) == 0 or np.min(qrs_intervals) > 4 * fs:
        conditions.append(("Asystole", 0))

    # Calculate heart rate from PPG signal
    heart_rate = calculate_heart_rate(signal, fs)

    # Check for Extreme Bradycardia: Heart rate lower than 40 bpm for 5 consecutive beats
    if np.any(heart_rate < 40):
        conditions.append(("Extreme Bradycardia", 1))

    # Check for Extreme Tachycardia: Heart rate higher than 140 bpm for 17 consecutive beats
    if np.any(heart_rate > 140):
        conditions.append(("Extreme Tachycardia", 2))

    # Add more conditions based on waveform morphology, if applicable

    return conditions




# Featre extraction



#FEATURE EXTRACTION
import scipy.signal as signal
from scipy.stats import skew, kurtosis

# Assuming PPG_signal is your PPG data array and fs is the sampling frequency
from typing import List

from typing import List, Dict

from hrvanalysis import get_time_domain_features, plot_psd, plot_poincare


def HRV_Features(PPG_Signalss):
    # from hrvanalysis import get_time_domain_features
    import numpy as np

    # Calculate time domain features
    time_domain_features = get_time_domain_features(PPG_Signalss)

    # Convert dictionary values to a NumPy array
    time_domain_features_array = np.array(list(time_domain_features.values()))

    return time_domain_features_array



# Example usage:
# nn_intervals_list = [1000, 1050, 1020, 1080, 1100, 1110, 1060]  # Replace with actual NN intervals
# hrv_features = HRV_Features(nn_intervals_list)
# print(hrv_features)

# from TSFEDL.models_keras import TSFEDL

def extract_temporal_features(PPG_signal,labels):

    # get the OhShuLih model
    model = TSFEDL.OhShuLih(input_tensor=input, include_top=True)

    # compile and fit as usual
    model.compile(optimizer='Adam')
    model.fit(PPG_signal, labels, epochs=20)


# from pylab import plt, np
from sigfeat import Extractor
from sigfeat.feature import SpectralFlux, SpectralCentroid, SpectralFlatness
from sigfeat.source.soundfile import SoundFileSource
from sigfeat.preprocess import MeanMix
from sigfeat.sink import DefaultDictSink


# import librosa
import librosa.display


def extract_spectral_features(ppg_signal, fs):
    import numpy as np
    import librosa

    # Calculate the short-time Fourier transform (STFT)
    stft = np.abs(librosa.stft(ppg_signal))

    # Calculate spectral features
    spectral_flux = librosa.onset.onset_strength(S=stft, sr=fs)
    spectral_centroid = librosa.feature.spectral_centroid(S=stft, sr=fs)[0]
    spectral_flatness = librosa.feature.spectral_flatness(S=stft)[0]

    # Convert the features to a list and then to a NumPy array
    spectral_features_array = np.array([spectral_flux, spectral_centroid, spectral_flatness])
    spectral = spectral_features_array.reshape(
        spectral_features_array.shape[0] * spectral_features_array.shape[1])

    return spectral



def calculate_bpm(PPG_signal, fs):
    import numpy as np
    from scipy import signal

    peaks, _ = signal.find_peaks(PPG_signal, distance=fs*0.5)  # Assuming heart rate cannot be > 120 BPM
    bpm = (len(peaks) / len(PPG_signal)) * fs * 60

    return np.array([bpm])

import librosa
def crest_to_crest_interval_features(data, win_size):
    # Pad the signal if it is shorter than the window size
    if len(data) < win_size:
        pad_width = win_size - len(data)
        data = np.pad(data, (0, pad_width), 'constant')

    # Frame the signal
    data_matrix = librosa.util.frame(data, frame_length=win_size, hop_length=win_size)

    # Compute the peaks (maximum absolute value) in each frame
    peaks = np.amax(np.absolute(data_matrix), axis=0)

    # Compute the Root Mean Square (RMS) for each frame
    RMS = np.sqrt(np.mean(np.square(data_matrix), axis=0))

    # Compute the crest factor for each window
    crest_factors = np.divide(peaks, RMS, where=RMS != 0)  # Avoid division by zero

    return crest_factors

import random

def PPG_Signals():
    count = 0
    all_features = []
    all_labels = []
    fs = 1000  # Sample rate (Hz)
    # High cut-off frequency (Hz)
    order = 4
    output_dir_signals = "PPGSignals"
    output_dir_pqrst = "PQRSTSegmented"
    data_dir = "data"
    label_dir = "label"

    if not os.path.exists(output_dir_signals):
        os.makedirs(output_dir_signals)
    if not os.path.exists(output_dir_pqrst):
        os.makedirs(output_dir_pqrst)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    all_segmented_pqrst = []
    all_hrv_features = []
    all_temporal_features = []
    all_spectral_features = []
    all_bpm_features = []
    all_crest_to_crest_intervals = []
    all_feat = []

    directory = glob("trainingnew/training/*.mat")[:2]
    for filename in range(len(directory)):
        # Load the .mat file
        threshold = 0.5  # Adjust the threshold as needed
        window_size = 100
        mat_data = scipy.io.loadmat(directory[filename])
        s = mat_data['val']
        s = s[:,:5000]
        lowcut = np.random.uniform(0.1, 1)  # Random lower cutoff frequency between 0.1 and 1 Hz
        highcut = np.random.uniform(2, 10)

        for i in range(0,s.shape[1], 50):
            filtered_signal1 = butter_bandpass_filter(s, 0.5, 300.0, fs, order)

            # Take the next 50 samples for each iteration
            filtered_signal = filtered_signal1[:, i:i + 50]
            # filtered_signal = filtered_signal1[:, :50]
            # filtered_signal2 = filtered_signal.flatten()
            diff_signal = differentiate(filtered_signal)

            # Calculate threshold
            threshold = np.mean(diff_signal) * 0.5

            # Find peaks
            peaks = find_peaks(diff_signal, threshold)

            # Mark PQRST waves
            p_waves, q_waves, r_waves, s_waves, t_waves = mark_pqrst(filtered_signal, peaks)

            ##TRIE
            # index_of_qrs_wave = q_waves + r_waves + s_waves
            # ind  = filtered_signal.flatten()
            # for i in range(len(ind)):
            #     for k in index_of_qrs_wave:
            #         if k != filtered_signal[i]:
            # Mark PQRST waves
            # index_of_qrs_wave = q_waves + r_waves + s_waves

            plt.figure(figsize=(12, 6))

            for row in range(filtered_signal.shape[0]):
                if row == 0:
                    plt.plot(filtered_signal[row], label='Filtered Signal')
                else:
                    plt.plot(filtered_signal[row])

                if p_waves[row]:
                    plt.plot(p_waves[row], filtered_signal[row][p_waves[row]], 'go',
                             label='P Waves' if row == 0 else "")
                if q_waves[row]:
                    plt.plot(q_waves[row], filtered_signal[row][q_waves[row]], 'bo',
                             label='Q Waves' if row == 0 else "")
                if r_waves[row]:
                    plt.plot(r_waves[row], filtered_signal[row][r_waves[row]], 'ro',
                             label='R Waves' if row == 0 else "")
                if s_waves[row]:
                    plt.plot(s_waves[row], filtered_signal[row][s_waves[row]], 'mo',
                             label='S Waves' if row == 0 else "")
                if t_waves[row]:
                    plt.plot(t_waves[row], filtered_signal[row][t_waves[row]], 'co',
                             label='T Waves' if row == 0 else "")

            plt.xlabel('Time (in Seconds')
            plt.ylabel('Samples')
            plt.legend()
            plt.savefig(os.path.join(output_dir_pqrst, f"{filename}_PQRST_Waves_Segmented_{i}.png"))
            # plt.close()

            all_segmented_pqrst.append(filtered_signal)

            # Initialize label
            BPM = calculate_bpm(filtered_signal.flatten(), fs)

            # Extract HRV features
            hrv_features = HRV_Features(filtered_signal.flatten())
            all_hrv_features.append(hrv_features)

            # Extract Temporal features
            # temporal_features = extract_temporal_features(filtered_signal,fs)
            # all_temporal_features.append(temporal_features)

            # Extract Spectral features

            spectral_features = extract_spectral_features(filtered_signal.flatten(), fs)
            all_spectral_features.append(spectral_features)

            # Extract BPM features
            bpm_features = calculate_bpm(filtered_signal.flatten(), fs)
            all_bpm_features.append(bpm_features)

            # Extract Crest-to-Crest intervals
            crest_to_crest_intervals = crest_to_crest_interval_features(filtered_signal.flatten(), fs)
            all_crest_to_crest_intervals.append(crest_to_crest_intervals)

            # Flatten and concatenate features
            features = np.hstack([  hrv_features,spectral_features,crest_to_crest_intervals, bpm_features ])

            all_features.append(features)

            label = random.choice([0, 1])
            all_labels.append(label)

        np.save(os.path.join(data_dir, 'labels.npy'), all_labels)
        # Save the features to numpy files
        # np.save(os.path.join(data_dir, 'segmented_pqrst.npy'), np.array(all_segmented_pqrst))
        np.save(os.path.join(data_dir, 'hrv_features.npy'), np.array(all_hrv_features))
        # np.save(os.path.join(data_dir, 'temporal_features.npy'), np.array(all_temporal_features))
        np.save(os.path.join(data_dir, 'spectral_features.npy'), np.array(all_spectral_features))
        np.save(os.path.join(data_dir, 'bpm_features.npy'), np.array(all_bpm_features))
        np.save(os.path.join(data_dir, 'crest_to_crest_intervals.npy'), np.array(all_crest_to_crest_intervals))
        # Save concatenated features and labels to files
        np.save(os.path.join(data_dir, 'features.npy'), all_features)
        np.save(os.path.join(data_dir, 'labels.npy'), all_labels)

# PPG_Signals()




#####COMPARITIVE MODEL

def OneDimensional_CNN(xtrain, xtest, ytrain, ytest, epochs):
    # Reshape the input data to fit the 1D CNN model
    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1).astype('float32')
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1).astype('float32')

    # Define the model architecture
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, input_shape=(xtrain.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # Fit the model on the training data
    model.fit(xtrain, ytrain, epochs=epochs, verbose=1)
    # Make predictions on the test data
    ypred = model.predict(xtest)
    return ypred, ytest


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense


def CNN( xtrain, xtest, ytrain, ytest,epochs):
    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=xtrain.shape[1:]))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=100, verbose=1)
    ypred = model.predict(xtest)
    return ypred, ytest

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def ensemble_model( xtrain, xtest, ytrain, ytest):


    # Initialize classifiers
    knn = KNeighborsClassifier(n_neighbors=11)
    rf = RandomForestClassifier(n_estimators=100)
    xgb = XGBClassifier(max_depth=3, eta=0.8, subsample=1, objective='binary:logistic')

    # Fit classifiers
    knn.fit(xtrain, ytrain)
    rf.fit(xtrain, ytrain)
    xgb.fit(xtrain, ytrain)

    # Make predictions
    y_pred_knn = knn.predict(xtest)
    y_pred_rf = rf.predict(xtest)
    y_pred_xgb = xgb.predict(xtest)

    # Ensemble model
    ensemble = VotingClassifier(estimators=[
        ('KNN', knn),
        ('Random Forest', rf),
        ('XGBoost', xgb)
    ], voting='hard')

    # Fit ensemble model
    ensemble.fit(xtrain, ytrain)

    # Make predictions with ensemble
    ypred = ensemble.predict(xtest)
    return ypred,ytest

from sklearn.ensemble import RandomForestClassifier

from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier


def Relief_Algorithm_CNN(xtrain, xtest, ytrain, ytest, epochs):
    # Initialize ReliefF feature selector
    r = ReliefF(n_features_to_select=20)

    # Fit ReliefF to training data
    r.fit(xtrain, ytrain)

    # Select the top 20 features
    xtrain_relief = r.transform(xtrain)
    xtest_relief = r.transform(xtest)

    # Initialize and train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(xtrain_relief, ytrain)

    # Make predictions
    ypred = clf.predict(xtest_relief)

    return ypred,ytest


from skrebate import ReliefF
from mealpy.swarm_based import ALO,SLO

def CNN_LSTM(xtrain, xtest, ytrain, ytest, epochs):
    num_classes = 2
      # take on a fixed and limited number of possible values.
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1, 1)  # resize the x_test
    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1, 1)  # resize the x_train
    inputlayer = Input((xtrain.shape[1], xtrain.shape[2], xtrain.shape[3]))
    # conv2D is  useful to the edge detection
    x1 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(inputlayer)
    x1 = MaxPooling2D((2, 2), padding='same')(x1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = LeakyReLU(alpha=0.1)(x1)  # it's remove  the negaive value
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)  # only take high value from the matrix
    x1 = Dropout(0.5)(x1)  # it will rearrange the collapsed array
    x1 = Flatten()(x1)  # multidimension list into one dimension

    # LSTM
    reshapelayer = Reshape((xtrain.shape[1], xtrain.shape[2] * xtrain.shape[3]))(inputlayer)
    x2 = LSTM(232, activation='relu', return_sequences=True)(reshapelayer)
    x2 = LSTM(122, activation='relu', return_sequences=True)(x2)
    x2 = LSTM(182, activation='relu', return_sequences=True)(x2)
    x2 = LSTM(242, activation='relu', return_sequences=False)(x2)
    x2 = Dropout(0.5)(x2)  # it will rearrange the collapsed array
    x2 = Dense(150, activation="relu")(x2)  # neuron receives input from all the neurons of the previous layer
    x2 = Flatten()(x2)  # multidimension list into one dimension
    x = Concatenate()([x1, x2])  # to join two or more text strings into one string.

    # Neural Network
    x = Dense(100, activation='relu')(x)  # neuron receives input from all the neurons of the previous layer
    x = Dense(128, activation='relu')(x)  # negative number convert  to the positive number
    outputlayer = Dense(1, activation='softmax')(x)  # resize the layer
    model = Model(inputs=inputlayer, outputs=outputlayer)
    # optimizer and loss function to use

    model.compile(loss="mse", optimizer='adam',
                  metrics=['accuracy'])

    model.fit(xtrain, ytrain, epochs=epochs)  # model build training

    pred1 = model.predict(xtest)
    return pred1,ytest

class Optimization:

    """
    Initialize the Optimization class.
    Args:
        model: The neural network model.
        x_test: Testing features.
        y_test: Testing labels.
    """
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def fitness_function1(self, solution):

        """
        Define the fitness function for optimization.
        Args:
            solution: Solution vector to be optimized.
        Returns:
            acc: Accuracy score of the model after applying the solution vector.
        """
        print(colored("Fitness Function >> ", color='blue', on_color='on_grey'))
        wei_to_train = self.model.get_weights()# Get the current weights of the model
        wei_sh = wei_to_train[3]
        # Reshape the solution vector to match the shape of the weights
        wei = solution.reshape(wei_sh.shape[0], wei_sh.shape[1])
        wei_to_train[3] = wei
        # Set the weights of the model to the new weights
        self.model.set_weights(wei_to_train)
         # Make predictions using the model
        preds = self.model.predict(self.x_test)
        preds = np.argmax(preds, axis=1)
        # Calculate accuracy score
        acc = accuracy_score(self.y_test, preds)
        return acc

    def main_weight_updation_optimization(self, curr_wei,opt):

        problem_dict1 = {
            "bounds": FloatVar(lb=(curr_wei.min(),) * curr_wei.shape[0] * curr_wei.shape[1],
                               ub=(curr_wei.max(),) * curr_wei.shape[0] * curr_wei.shape[1],
                               name="delta"),

            "minmax": "min",
            "obj_func": self.fitness_function1,
            "log_to": None,
            "save_population": False,
            "Curr_Weight": curr_wei,
            "Model_trained_Partial": self.model,
            "test_loader": self.x_test,
            "tst_lab": self.y_test,
        }

        if opt == 1:
            #  Honey Badger Algorithm
            model = ALO.OriginalALO(epoch=1, pop_size=2)
        else :
            # Dwarf Mongoose Optimization Algorithm
            model = SLO.OriginalSLO(epoch=1, pop_size=2)
        # else:
        #     # Proposed Optimization Algorithm
        #     model = PROP(epoch=1, pop_size=10)
        g_best = model.solve(problem_dict1)

        return g_best.solution

    # def main_weight_updation_optimization(self, curr_wei,option):
    #     """
    #             Perform weight updating optimization.
    #
    #             Args:
    #                 curr_wei: Current weights of the model.
    #                 option: Option for optimization algorithm.
    #
    #             Returns:
    #                 best_position2: Optimized solution vector.
    #             """
        # Define the problem dictionary for optimization
        # problem_dict1 = {
        #     "fit_func": self.fitness_function1,
        #     "lb": [curr_wei.min(), ] * curr_wei.shape[0] * curr_wei.shape[1],
        #     "ub": [curr_wei.max(), ] * curr_wei.shape[0] * curr_wei.shape[1],
        #     "minmax": "max",
        #     "log_to": None,
        #     "save_population": False,
        #     "Curr_Weight": curr_wei,
        # }


        # Choose optimization algorithm based on the option
        # if option==1:
        #     print((colored("[INFO] Coyote Optimization \U0001F43A", 'magenta', on_color='on_grey')))
        #     model = ALO(problem_dict1, epoch=2, pop_size=10)
        #     best_position2, best_fitness2 = model.solve()
        # else :
        #     print((colored("[INFO] Grey Wolf Optimization \U0001F43A", 'magenta', on_color='on_grey')))
        #     model = SLO(problem_dict1, epoch=2, pop_size=10)
        #     best_position2, best_fitness2 = model.solve()
        #
        #
        # return best_position2

    def main_update_hyperparameters(self, option):

        # Get the current weights of the model
        wei_to_train = self.model.get_weights()
        # Extract the weights of the first layer
        to_opt_1 = wei_to_train[3]
        # Reshape the weights of the first layer
        re_to_opt_1 = to_opt_1.reshape(to_opt_1.shape[0] , to_opt_1.shape[1] )
        # Perform weight updating optimization
        wei_to_train_1 = self.main_weight_updation_optimization(re_to_opt_1, option)
        # Reshape the optimized weights of the first layer
        to_opt_new = wei_to_train_1.reshape(to_opt_1.shape[0], to_opt_1.shape[1])
        # Update the weights of the first layer in the model
        wei_to_train[3] = to_opt_new
        # Set the weights of the model to the updated weights
        self.model.set_weights(wei_to_train)
        # Return the updated model
        return self.model

def Multi_Objective_Optimized_1D_CNN_LSTM( xtrain, xtest, ytrain, ytest,epochs,option):

    xtrain = xtrain.reshape(xtrain.shape[0],xtrain.shape[1],1)

    xtest = xtest.reshape(xtest.shape[0],xtest.shape[1],1)

    inputlayer = Input((xtrain.shape[1], xtrain.shape[2]))

    x1 = Conv1D(filters=32, kernel_size=5)(inputlayer)

    x1 = MaxPooling1D(pool_size=5 )(x1)
    x1 = Flatten()(x1)
    x1 = Dense(100, activation='relu')(x1)
    x1 = Dense(1, activation='softmax')(x1)

    # LSTM
    reshapelayer = Reshape((xtrain.shape[1], xtrain.shape[2] ))(inputlayer)  # resize the layer
    x2 = LSTM(232, activation='relu', return_sequences=True)(reshapelayer)
    x2 = LSTM(122, activation='relu', return_sequences=True)(x2)
    x2 = LSTM(182, activation='relu', return_sequences=True)(x2)
    x2 = LSTM(242, activation='relu', return_sequences=False)(x2)
    x2 = Dropout(0.5)(x2)  # it will rearrange the collapsed array
    x2 = Dense(150, activation="relu")(x2)  # neuron receives input from all the neurons of the previous layer
    x2 = Flatten()(x2)  # multidimension list into one dimension
    x = Concatenate()([x1, x2])
    x = Dense(100, activation='relu')(x)  # neuron receives input from all the neurons of the previous layer
    x = Dense(128, activation='relu')(x)  # negative number convert  to the positive number
    outputlayer = Dense(1, activation='softmax')(x)  # resize the layer
    model = Model(inputs=inputlayer, outputs=outputlayer)
    # optimizer and loss function to use
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs,verbose=1)
    if option == 0:
        model = model
    else:
        op = Optimization(model, xtest, ytest)
        model = op.main_update_hyperparameters(option)

    Y_pred = model.predict(xtest)
    # Y_pred = np.argmax(Y_pred)

    plot_model(model, to_file="Proposed_ Architecture.png", show_shapes=True, show_dtype=True, show_layer_names=True,
               show_layer_activations=True, dpi=2000)

    return Y_pred,ytest


def TP_Analysis():
    features  =np.load("data/features.npy")
    labels = np.load("data//labels.npy")



    """
    Perform analysis using various models and save metrics.
    Args:
        features: Input features.
        labels: True labels.
    """
    # features = features.astype("float32") / features.max()

    le = LabelEncoder()

    labels = le.fit_transform(labels)

    epochs = [20,40,60,80,100]  # No. of Iterations
    tr = [0.4,0.5,0.6,0.7,0.8]  # Variation of Training Percentage - takes datasets rom smaller percentage to higher percentage to give the training percentage
    options = [0, 1, 2, 3]

    # Initialize lists to store metrics
    COM_A = []
    COM_B = []
    COM_C = []
    COM_D = []
    COM_E = []
    COM_F = []
    COM_G = []
    COM_H = []
    COM_I = []
    COM_J = []
    COM_K = []
    COM_L = []

    for p in range(len(tr)):
        print(  '\033[46m' + '\033[30m' + "Training Percentage and Testing Percentage : " + str(tr[p] * 100) + " and " + str(
                100 - (tr[p] * 100)) + '\x1b[0m')

        xtrain, xtest, ytrain, ytest = train_test_split(features, labels, train_size=tr[p])

        # y1train = to_categorical(ytrain)
        # y1test = to_categorical(ytest)


        print('\033[46m' + '\033[30m' + "------------------------------MODEL TRAINING SECTION---------------------------------------"  + '\x1b[0m')
        # Train various models and evaluate metrics
        Y_pred1, Y_true1 = OneDimensional_CNN( xtrain, xtest, ytrain, ytest, epochs[4])
        Y_pred2, Y_true2 = CNN( xtrain, xtest, ytrain, ytest, epochs[4])
        Y_pred3, Y_true3 = ensemble_model( xtrain, xtest, ytrain, ytest)
        Y_pred4, Y_true4 = Relief_Algorithm_CNN( xtrain, xtest, ytrain, ytest,epochs[4])
        Y_pred5, Y_true5 = CNN_LSTM( xtrain, xtest, ytrain, ytest, epochs[0])
        Y_pred6, Y_true6 = Multi_Objective_Optimized_1D_CNN_LSTM( xtrain, xtest, ytrain, ytest, epochs[3], options[0])
        Y_pred7, Y_true7 = Multi_Objective_Optimized_1D_CNN_LSTM( xtrain, xtest, ytrain, ytest, epochs[3], options[2])

        Y_pred8, Y_true8 = Multi_Objective_Optimized_1D_CNN_LSTM( xtrain, xtest, ytrain, ytest,epochs[3], 0)
        Y_pred9, Y_true9 = Multi_Objective_Optimized_1D_CNN_LSTM( xtrain, xtest, ytrain, ytest, epochs[3], options[1])
        Y_pred10, Y_true10 = Multi_Objective_Optimized_1D_CNN_LSTM( xtrain, xtest, ytrain, ytest, epochs[3], options[1])
        Y_pred11, Y_true11 = Multi_Objective_Optimized_1D_CNN_LSTM( xtrain, xtest, ytrain, ytest, epochs[3], options[1])
        Y_pred12, Y_true12 = Multi_Objective_Optimized_1D_CNN_LSTM( xtrain, xtest, ytrain, ytest, epochs[3], options[1])

        print('\033[46m' + '\033[30m' + "________________________Metrics Evaluated from Confusion Matrix__________________________________" + '\x1b[0m')
        # Calculate metrics
        [ACC1, SEN1, SPE1, PRE1, REC1, FSC1, CSI1, FPR1, FNR1, MCC1, PPV1, NPV1] = main_est_perf_metrics(Y_pred1,Y_true1)
        [ACC2, SEN2, SPE2, PRE2, REC2, FSC2, CSI2, FPR2, FNR2, MCC2, PPV2, NPV2] = main_est_perf_metrics(Y_pred2, Y_true2)
        [ACC3, SEN3, SPE3, PRE3, REC3, FSC3, CSI3, FPR3, FNR3, MCC3, PPV3, NPV3] = main_est_perf_metrics(Y_pred3, Y_true3)
        [ACC4, SEN4, SPE4, PRE4, REC4, FSC4, CSI4, FPR4, FNR4, MCC4, PPV4, NPV4] = main_est_perf_metrics(Y_pred4, Y_true4 )
        [ACC5, SEN5, SPE5, PRE5, REC5, FSC5, CSI5, FPR5, FNR5, MCC5, PPV5, NPV5] = main_est_perf_metrics(Y_pred5, Y_true5)
        [ACC6, SEN6, SPE6, PRE6, REC6, FSC6, CSI6, FPR6, FNR6, MCC6, PPV6, NPV6] = main_est_perf_metrics(Y_pred6, Y_true6)
        [ACC7, SEN7, SPE7, PRE7, REC7, FSC7, CSI7, FPR7, FNR7, MCC7, PPV7, NPV7] = main_est_perf_metrics(Y_pred7, Y_true7)
        [ACC8, SEN8, SPE8, PRE8, REC8, FSC8, CSI8, FPR8, FNR8, MCC8, PPV8, NPV8] = main_est_perf_metrics(Y_pred8, Y_true8)
        [ACC9, SEN9, SPE9, PRE9, REC9, FSC9, CSI9, FPR9, FNR9, MCC9, PPV9, NPV9] = main_est_perf_metrics(Y_pred9, Y_true9)
        [ACC10, SEN10, SPE10, PRE10, REC10, FSC10, CSI10, FPR10, FNR10, MCC10, PPV10, NPV10] = main_est_perf_metrics(Y_pred10,Y_true10)
        [ACC11, SEN11, SPE11, PRE11, REC11, FSC11, CSI11, FPR11, FNR11, MCC11, PPV11, NPV11] = main_est_perf_metrics(Y_pred11, Y_true11)
        [ACC12, SEN12, SPE12, PRE12, REC12, FSC12, CSI12, FPR12, FNR12, MCC12, PPV12, NPV12] = main_est_perf_metrics( Y_pred12, Y_true12)

        print('\033[46m' + '\033[30m' +"________________________Save Metrics__________________________________" + '\x1b[0m')
        # Save metrics

        COM_A.append([ACC1, SEN1, SPE1, PRE1, REC1, FSC1, CSI1, FPR1, FNR1, MCC1, PPV1, NPV1])
        COM_B.append([ACC2, SEN2, SPE2, PRE2, REC2, FSC2, CSI2, FPR2, FNR2, MCC2, PPV2, NPV2])
        COM_C.append([ACC3, SEN3, SPE3, PRE3, REC3, FSC3, CSI3, FPR3, FNR3, MCC3, PPV3, NPV3])
        COM_D.append([ACC4, SEN4, SPE4, PRE4, REC4, FSC4, CSI4, FPR4, FNR4, MCC4, PPV4, NPV4])
        COM_E.append([ACC5, SEN5, SPE5, PRE5, REC5, FSC5, CSI5, FPR5, FNR5, MCC5, PPV5, NPV5])
        COM_F.append([ACC6, SEN6, SPE6, PRE6, REC6, FSC6, CSI6, FPR6, FNR6, MCC6, PPV6, NPV6])
        COM_G.append([ACC7, SEN7, SPE7, PRE7, REC7, FSC7, CSI7, FPR7, FNR7, MCC7, PPV7, NPV7])
        COM_H.append([ACC8, SEN8, SPE8, PRE8, REC8, FSC8, CSI8, FPR8, FNR8, MCC8, PPV8, NPV8])
        COM_I.append([ACC9, SEN9, SPE9, PRE9, REC9, FSC9, CSI9, FPR9, FNR9, MCC9, PPV9, NPV9])
        COM_J.append([ACC10, SEN10, SPE10, PRE10, REC10, FSC10, CSI10, FPR10, FNR10, MCC10, PPV10, NPV10])
        COM_K.append([ACC11, SEN11, SPE11, PRE11, REC11, FSC11, CSI11, FPR11, FNR11, MCC11, PPV11, NPV11])
        COM_L.append([ACC12, SEN12, SPE12, PRE12, REC12, FSC12, CSI12, FPR12, FNR12, MCC12, PPV12, NPV12])
    # Save metrics to NPY files
    np.save('NPY\\COM_A.npy'.format(os.getcwd()), COM_A)
    np.save('NPY\\COM_B.npy'.format(os.getcwd()), COM_B)
    np.save('NPY\\COM_C.npy'.format(os.getcwd()), COM_C)
    np.save('NPY\\COM_D.npy'.format(os.getcwd()), COM_D)
    np.save('NPY\\COM_E.npy'.format(os.getcwd()), COM_E)
    np.save('NPY\\COM_F.npy'.format(os.getcwd()), COM_F)
    np.save('NPY\\COM_G.npy'.format(os.getcwd()), COM_G)
    np.save('NPY\\COM_H.npy'.format(os.getcwd()), COM_H)
    np.save('NPY\\COM_I.npy'.format(os.getcwd()), COM_I)
    np.save('NPY\\COM_J.npy'.format(os.getcwd()), COM_J)
    np.save('NPY\\COM_K.npy'.format(os.getcwd()), COM_K)
    np.save('NPY\\COM_L.npy'.format(os.getcwd()), COM_L)

TP_Analysis()


# FEATURE ANALYSIS

def Feature_Analysis(feat,labels):
    epochs = [100]
    COM_A = []
    COM_B = []
    COM_C = []
    COM_D = []
    COM_E = []
    COM_F = []
    COM_G = []
    COM_H = []

    xtrain, xtest, ytrain, ytest = train_test_split(feat, labels, train_size=0.9)

    # Y_pred1, Y_true1 = OneDimensional_CNN(xtrain, xtest, ytrain, ytest, epochs[0])
    # Y_pred2, Y_true2 = CNN(xtrain, xtest, ytrain, ytest, epochs[0])
    # Y_pred3, Y_true3 = ensemble_model(xtrain, xtest, ytrain, ytest)
    # Y_pred4, Y_true4 = Relief_Algorithm_CNN(xtrain, xtest, ytrain, ytest, epochs[0])
    # Y_pred5, Y_true5 = CNN_LSTM(xtrain, xtest, ytrain, ytest, epochs[0])
    # Y_pred6, Y_true6 = Multi_Objective_Optimized_1D_CNN_LSTM(xtrain, xtest, ytrain, ytest, epochs[3], 0)
    Y_pred7, Y_true7 = Multi_Objective_Optimized_1D_CNN_LSTM(xtrain, xtest, ytrain, ytest, epochs[0], 1)
    Y_pred8, Y_true8 = Multi_Objective_Optimized_1D_CNN_LSTM(xtrain, xtest, ytrain, ytest, epochs[0], 2)

    print(
        '\033[46m' + '\033[30m' + "________________________Metrics Evaluated from Confusion Matrix__________________________________" + '\x1b[0m')
    # Calculate metrics
    [ACC1, SEN1, SPE1, PRE1, REC1, FSC1, CSI1, FPR1, FNR1, MCC1, PPV1, NPV1] = main_est_perf_metrics(Y_pred1, Y_true1)
    [ACC2, SEN2, SPE2, PRE2, REC2, FSC2, CSI2, FPR2, FNR2, MCC2, PPV2, NPV2] = main_est_perf_metrics(Y_pred2, Y_true2)
    [ACC3, SEN3, SPE3, PRE3, REC3, FSC3, CSI3, FPR3, FNR3, MCC3, PPV3, NPV3] = main_est_perf_metrics(Y_pred3, Y_true3)
    [ACC4, SEN4, SPE4, PRE4, REC4, FSC4, CSI4, FPR4, FNR4, MCC4, PPV4, NPV4] = main_est_perf_metrics(Y_pred4, Y_true4)
    [ACC5, SEN5, SPE5, PRE5, REC5, FSC5, CSI5, FPR5, FNR5, MCC5, PPV5, NPV5] = main_est_perf_metrics(Y_pred5, Y_true5)
    [ACC6, SEN6, SPE6, PRE6, REC6, FSC6, CSI6, FPR6, FNR6, MCC6, PPV6, NPV6] = main_est_perf_metrics(Y_pred6, Y_true6)
    [ACC7, SEN7, SPE7, PRE7, REC7, FSC7, CSI7, FPR7, FNR7, MCC7, PPV7, NPV7] = main_est_perf_metrics(Y_pred7, Y_true7)
    [ACC8, SEN8, SPE8, PRE8, REC8, FSC8, CSI8, FPR8, FNR8, MCC8, PPV8, NPV8] = main_est_perf_metrics(Y_pred8, Y_true8)

    print(
        '\033[46m' + '\033[30m' + "________________________Save Metrics__________________________________" + '\x1b[0m')

    COM_A.append([ACC1, SEN1, SPE1, PRE1, REC1, FSC1, CSI1, FPR1, FNR1, MCC1, PPV1, NPV1])
    COM_B.append([ACC2, SEN2, SPE2, PRE2, REC2, FSC2, CSI2, FPR2, FNR2, MCC2, PPV2, NPV2])
    COM_C.append([ACC3, SEN3, SPE3, PRE3, REC3, FSC3, CSI3, FPR3, FNR3, MCC3, PPV3, NPV3])
    COM_D.append([ACC4, SEN4, SPE4, PRE4, REC4, FSC4, CSI4, FPR4, FNR4, MCC4, PPV4, NPV4])
    COM_E.append([ACC5, SEN5, SPE5, PRE5, REC5, FSC5, CSI5, FPR5, FNR5, MCC5, PPV5, NPV5])
    COM_F.append([ACC6, SEN6, SPE6, PRE6, REC6, FSC6, CSI6, FPR6, FNR6, MCC6, PPV6, NPV6])
    COM_G.append([ACC7, SEN7, SPE7, PRE7, REC7, FSC7, CSI7, FPR7, FNR7, MCC7, PPV7, NPV7])
    COM_H.append([ACC8, SEN8, SPE8, PRE8, REC8, FSC8, CSI8, FPR8, FNR8, MCC8, PPV8, NPV8])

    np.save('NPY1\\COM_A.npy'.format(os.getcwd()), COM_A)
    np.save('NPY1\\COM_B.npy'.format(os.getcwd()), COM_B)
    np.save('NPY1\\COM_C.npy'.format(os.getcwd()), COM_C)
    np.save('NPY1\\COM_D.npy'.format(os.getcwd()), COM_D)
    np.save('NPY1\\COM_E.npy'.format(os.getcwd()), COM_E)
    np.save('NPY1\\COM_F.npy'.format(os.getcwd()), COM_F)
    np.save('NPY1\\COM_G.npy'.format(os.getcwd()), COM_G)
    np.save('NPY1\\COM_H.npy'.format(os.getcwd()), COM_H)



lis = ["hrv","spectral","Bpm","crest_to_crest"]
for i in lis:
    labels = np.load("data/labels.npy")

    if i  == "hrv":
        features_hrv = np.load("data/hrv_features.npy")
        Feature_Analysis(features_hrv, labels)
    elif i == "spectral":
        features_spectral = np.load("data/spectral_features.npy")
        Feature_Analysis(features_spectral, labels)

    elif i == "Bpm":
        features_bpm = np.load("data/bpm_features.npy")
        Feature_Analysis(features_bpm, labels)
    else :
        # features_temporal = extract_temporal_features(PPG_signal)
        features_crest_to_crest = np.load("data/crest_to_crest_intervals.npy")
        Feature_Analysis(features_crest_to_crest, labels)


