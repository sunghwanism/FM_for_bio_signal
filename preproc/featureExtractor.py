import os

import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

def load_ecg(df, window_size=30):

    fs = 256

    # Standardize ECG data
    ecg_scaler = StandardScaler()
    df['ECG'] = ecg_scaler.fit_transform(df['ECG'].values.reshape(-1, 1))

    data = []
    for idx in range(len(df) // (fs * window_size)):
        start_window = window_size * idx * fs
        
        ecg = df['ECG'].iloc[start_window:start_window + window_size * fs].values
        labels = df['psg_status'].iloc[start_window]
        if labels in [2, 3]:
            labels = 2
        elif labels in [4, 5]:
            labels = 3


        data.append((ecg, labels))

    return data


# Define a new function
def my_processing(ecg_signal,fs):
    # Do processing
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs, method="neurokit")
    instant_peaks, rpeaks, = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
    rate = nk.ecg_rate(rpeaks, sampling_rate=fs, desired_length=len(ecg_cleaned))
    quality = nk.ecg_quality(ecg_cleaned, sampling_rate=fs)
    edr = nk.ecg_rsp(rate, sampling_rate=fs)

    # Prepare output
    signals = pd.DataFrame({"ECG_Raw": ecg_signal,
                            "ECG_Clean": ecg_cleaned,
                            "ECG_Rate": rate,
                            "ECG_Quality": quality,
                            "EDR": edr})
    signals = pd.concat([signals, instant_peaks], axis=1)
    info = rpeaks

    return signals, info


def get_ecg_features(subject,load_dir, save_dir):

    data_path = os.path.join(load_dir, f'subject_{subject}_ecg.csv')
    test_data = pd.read_csv(data_path)
    
    # data = load_ecg(test_data, window_size=30)

    fs = 256
    test_ecg = test_data.ECG

    ecg_proc, info = my_processing(test_ecg, fs)
    
    #epoch data  
    window = 30 
    intervals = np.arange(0, len(ecg_proc)-1, fs*window)
    epochs = nk.epochs_create(ecg_proc, events=intervals, sampling_rate=fs)

    epochs_df = nk.epochs_to_df(epochs)
    epochs_df = epochs_df[0:len(ecg_proc)] #fix issue with output size

    feat = nk.ecg_intervalrelated(epochs)

    #fix issues with output being in list

    columns_to_unbracket = feat.columns[2:]

    for col in columns_to_unbracket:
        feat[col] = feat[col].apply(lambda x: x[0][0])

    #labels 
    labels = test_data.psg_status.to_frame()
    epochs_df['labels'] = labels['psg_status']

    # binary classification
    epochs_df['labels2'] = labels['psg_status'].replace([1, 2, 3, 4, 5], 1)

    # 4 classes
    epochs_df['labels4'] = labels['psg_status'].replace([1,2], 1)
    epochs_df['labels4'] = epochs_df['labels4'].replace([3,4], 2)
    epochs_df['labels4'] = epochs_df['labels4'].replace([5], 3)
    
    #same for features 
    labels_trunc = labels.iloc[::256*30, :].reset_index()


    feat['labels'] = labels_trunc['psg_status'].values

    feat['labels2'] = feat['labels'].replace([1, 2, 3, 4, 5], 1).values

    feat['labels4'] = feat['labels'].replace([1,2], 1).values
    feat['labels4'] = feat['labels4'].replace([3,4], 2).values
    feat['labels4'] = feat['labels4'].replace([5], 3).values

    # print(feat.head())
    # print(epochs_df.head())
    
    feat.to_csv(os.path.join(save_dir,'subject_'+subject+'_ecg_feat.csv'))
    epochs_df.to_csv(os.path.join(save_dir,'subject_'+subject+'_ecg_proc.csv'))

def subject_extractor(data_dir):
    file_list = os.listdir(data_dir)
    result = []
    file_list = [file for file in file_list if 'ecg' in file]
    
    for file in file_list:
        subject_id = file.split('_')[1]
        result.append(subject_id)
        
    return result


def get_ecg_features_np(subject, load_dir, save_dir):
    
    test_ecg = np.load(os.path.join(load_dir, f'{subject}_ecg.npy'))
    test_ecg = test_ecg.squeeze().reshape(-1)
    
    fs = 256

    ecg_proc, info = my_processing(test_ecg, fs)
    
    #epoch data  
    window = 30 
    intervals = np.arange(0, len(ecg_proc)-1, fs*window)
    epochs = nk.epochs_create(ecg_proc, events=intervals, sampling_rate=fs)

    epochs_df = nk.epochs_to_df(epochs)
    epochs_df = epochs_df[0:len(ecg_proc)] #fix issue with output size

    feat = nk.ecg_intervalrelated(epochs)

    #fix issues with output being in list

    columns_to_unbracket = feat.columns[2:]

    for col in columns_to_unbracket:
        feat[col] = feat[col].apply(lambda x: x[0][0])

    #labels 
    labels = np.load(os.path.join(load_dir, f'{subject}_labeled_sleep.npy'))
    labels = pd.DataFrame(labels, columns=['psg_status'])
    epochs_df['labels'] = labels['psg_status']

    # binary classification
    epochs_df['labels2'] = labels['psg_status'].replace([1, 2, 3, 4, 5], 1)

    # 4 classes
    epochs_df['labels4'] = labels['psg_status'].replace([1,2], 1)
    epochs_df['labels4'] = epochs_df['labels4'].replace([3,4], 2)
    epochs_df['labels4'] = epochs_df['labels4'].replace([5], 3)
    
    #same for features 
    labels_trunc = labels.iloc[::256*30, :].reset_index()


    feat['labels'] = labels_trunc['psg_status'].values

    feat['labels2'] = feat['labels'].replace([1, 2, 3, 4, 5], 1).values

    feat['labels4'] = feat['labels'].replace([1,2], 1).values
    feat['labels4'] = feat['labels4'].replace([3,4], 2).values
    feat['labels4'] = feat['labels4'].replace([5], 3).values

    # print(feat.head())
    # print(epochs_df.head())
    
    feat.to_csv(os.path.join(save_dir,'subject_'+subject+'_ecg_feat.csv'))
    epochs_df.to_csv(os.path.join(save_dir,'subject_'+subject+'_ecg_proc.csv'))
