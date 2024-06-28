import glob
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import neurokit2 as nk
import numpy as np
import heartpy as hp



class Read_Dataset(object):
    def __init__(self, db):
        self.db = db
        self.fs = 1000
        self.show = False

    def read_path(self):
        if self.db == 1:
            files = glob.glob("training\\training\\*.mat")
        return files

    def read_signal(self, show = False):
        fs = 360

        sig = scipy.io.loadmat(self.file)
        data = sig['val']
        time = np.arange(data.size) / fs

        if show:
            plt(time, data)
            plt.show()
        return data


    def butter_bandpass(self, lowcut=0.5, highcut=300, order=5):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data):
        b, a = self.butter_bandpass()
        # b, a = hp.filtering.butter_bandpass(0.5, 3000, 1000, order=2)

        y = filtfilt(b, a, data)
        if self.show:
            plt.plot(y)
            plt.show()
        return y

    def HRV_Features(self):
        results = {}

        hr = 60000 / self.ecg_signal

        # HRV metrics
        results['Mean RR (ms)'] = np.mean(self.ecg_signal)
        results['STD RR/SDNN (ms)'] = np.std(self.ecg_signal)
        results['Mean HR (Kubios\' style) (beats/min)'] = 60000 / np.mean(self.ecg_signal)
        results['Mean HR (beats/min)'] = np.mean(hr)
        results['STD HR (beats/min)'] = np.std(hr)
        results['Min HR (beats/min)'] = np.min(hr)
        results['Max HR (beats/min)'] = np.max(hr)
        results['RMSSD (ms)'] = np.sqrt(np.mean(np.square(np.diff(self.ecg_signal))))
        results['NN50'] = np.sum(np.abs(np.diff(self.ecg_signal)) > 50) * 1
        results['pNN50 (%)'] = 100 * np.sum((np.abs(np.diff(self.ecg_signal)) > 50) * 1) / len(self.ecg_signal)
        return results

    def get_BPM(self):
        # run analysis

        wd, m = hp.process(self.ecg_signal, self.fs)



        if self.show:
            # visualise in plot of custom size
            plt.figure(figsize=(12, 4))
            hp.plotter(wd, m)

            # display computed measures
            for measure in m.keys():
                print('%s: %f' % (measure, m[measure]))

        return


    def mark_pqrst1(self):
        for ind in range(self.ecg.shape[0]):
            self.ecg_signal = self.ecg[ind, :]
            _, rpeaks = nk.ecg_peaks(self.ecg_signal, sampling_rate=self.fs)
            _, waves_peak = nk.ecg_delineate(self.ecg_signal, rpeaks, sampling_rate=self.fs, method="peak")
            epochs = nk.ecg_segment(self.ecg_signal, rpeaks=None, sampling_rate=self.fs, show=True)
            self.get_BPM()

            if self.show:
                plot = nk.events_plot(rpeaks['ECG_R_Peaks'], self.ecg_signal)
                plot = nk.events_plot([waves_peak['ECG_T_Peaks'][:3],
                                       waves_peak['ECG_P_Peaks'][:3],
                                       waves_peak['ECG_Q_Peaks'][:3],
                                       waves_peak['ECG_S_Peaks'][:3]], self.ecg_signal[:4000])

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

    def PQRST_Extraction(self):
        threshold = np.mean(diff_signal) * 0.5

        # Find peaks
        peaks = find_peaks(diff_signal, threshold)

        # Mark PQRST waves
        p_waves, q_waves, r_waves, s_waves, t_waves = mark_pqrst(filtered_signal, peaks)




    def get_data(self):
        data  = self.read_path()
        for self.file in data:
            self.ecg = self.butter_bandpass_filter(self.read_signal())
            self.ecg = self.read_signal()
            self.mark_pqrst1()


    def __call__(self, *args, **kwargs):
        self.get_data()


file = (Read_Dataset(1))()


class  Feature_Extration():
    def __init__(self, db):
        self.db = db

    def Temporal_Features(self):
        pass

