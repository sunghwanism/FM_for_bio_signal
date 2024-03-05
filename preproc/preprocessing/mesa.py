import os
import csv
import glob

import numpy as np

import matplotlib.pyplot as plt
import numpy as np

from source import utils
from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject import Subject
from source.constants import Constants
from source.mesa.mesa_actigraphy_service import MesaActigraphyService
from source.mesa.mesa_heart_rate_service import MesaHeartRateService
from source.mesa.mesa_ecg_service import MesaECGService
from source.mesa.mesa_psg_service import MesaPSGService
from source.mesa.mesa_time_based_service import MesaTimeBasedService

from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.epoch import Epoch
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.preprocessing.heart_rate.heart_rate_service import HeartRateService
from source.preprocessing.ecg.ecg_feature_service import ECGFeatureService
from source.preprocessing.ecg.ecg_service import ECGService
from source.preprocessing.interval import Interval
from source.preprocessing.time.time_based_feature_service import TimeBasedFeatureService

def get_all_files():
    # project_root = str(utils.get_project_root())
    project_root = "/volumes/990pro_2TB/UofT/Intro_DL/project/data/mesa"
    return glob.glob(project_root + "/polysomnography/edfs/*edf")


def get_all_subjects():
    all_files = get_all_files()
    all_subjects = []
    for file in all_files:
        file_id = file[-8:-4]
        subject = MesaSubjectBuilder.build(file_id)
        if subject is not None:
            all_subjects.append(subject)

    return all_subjects


class MesaSubjectBuilder(object):

    @staticmethod
    def build(file_id):
        if Constants.VERBOSE:
            print('Building MESA subject ' + file_id + '...')

        raw_labeled_sleep = MesaPSGService.load_raw(file_id) # annotation sleep -nsrr.xml
        print("Finish load sleep stage", raw_labeled_sleep.shape)
        heart_rate_collection = MesaHeartRateService.load_raw(file_id) # edf file // 1Hz
        print("Finish load raw HR", heart_rate_collection.data.shape)
        activity_count_collection = MesaActigraphyService.load_raw(file_id)
        print("Finish load raw Actigraphy", activity_count_collection.data.shape)
        ecg_collection = MesaECGService.load_raw(file_id)
        print("Finish Load raw ECG", ecg_collection.data.shape)
        

        if activity_count_collection.data[0][0] != -1 : # and circadian_model is not None:

            interval = Interval(start_time=0, end_time=np.shape(raw_labeled_sleep)[0]) # len=43170

            activity_count_collection = ActivityCountService.crop(activity_count_collection, interval)
            heart_rate_collection = HeartRateService.crop(heart_rate_collection, interval)
            ecg_collection = ECGService.crop(ecg_collection, interval)

            valid_epochs = []

            for timestamp in range(interval.start_time, interval.end_time, Epoch.DURATION):
                epoch = Epoch(timestamp=timestamp, index=len(valid_epochs))
                                
                activity_count_indices = ActivityCountFeatureService.window_epoch(activity_count_collection.timestamps, epoch)
                heart_rate_indices = HeartRateFeatureService.window_epoch(heart_rate_collection.timestamps, epoch)
                
                ecg_indices = ECGFeatureService.window_epoch(ecg_collection.timestamps, epoch)
                

                if len(activity_count_indices) > 0 and 0 not in heart_rate_collection.values[heart_rate_indices] and 0 not in ecg_collection.values[ecg_indices]:
                    valid_epochs.append(epoch)

                else:
                    pass

            labeled_sleep = np.expand_dims(MesaPSGService.crop(psg_labels=raw_labeled_sleep,
                                                               valid_epochs=valid_epochs), axis=1)
            print("Final labeled_sleep", labeled_sleep.shape)

            feature_count = ActivityCountFeatureService.build_from_collection(activity_count_collection, valid_epochs)
            print("Final Activity count_features", np.array(feature_count).shape)
            
            feature_hr = HeartRateFeatureService.build_from_collection(heart_rate_collection, valid_epochs)
            print("Final hr_features", np.array(feature_hr).shape)
            
            feature_ecg = ECGFeatureService.build_from_collection(ecg_collection, valid_epochs)
            print("Final ecg_features", np.array(feature_ecg).shape)
            
            
            feature_dictionary = {FeatureType.count: feature_count,
                                  FeatureType.heart_rate: feature_hr,
                                  FeatureType.ecg: feature_ecg}

            subject = Subject(subject_id=file_id,
                              labeled_sleep=labeled_sleep,
                              feature_dictionary=feature_dictionary)


            return subject


def mesa_preprocessing(subject_set, savepath="/NFS/Users/moonsh/data/mesa/preproc/npy/"):
    Error = []
    for subject in subject_set:

        try:
            subject_info = MesaSubjectBuilder.build(subject)
            
            activity_count = subject_info.feature_dictionary[FeatureType.count]
            heart_rate = subject_info.feature_dictionary[FeatureType.heart_rate]
            labeled_sleep = subject_info.labeled_sleep.squeeze()
            ecg = subject_info.feature_dictionary[FeatureType.ecg]
            
            print("** Finish Feature Preprocessing **")

            np.save(savepath+subject+"_activity_count.npy", activity_count)
            np.save(savepath+subject+"_heart_rate.npy", heart_rate)
            np.save(savepath+subject+"_labeled_sleep.npy", labeled_sleep)
            np.save(savepath+subject+"_ecg.npy", ecg)
            
            # np.save("../output/"+subject+"_cosine.npy", cosine)

            print("Activity", activity_count.shape, "HR", heart_rate.shape, "Labeled", labeled_sleep.shape, "ECG", ecg.shape)
            
        except:
            print("Error in subject", subject)
            Error.append(subject)
            pass
        
        print("########################"*3)
        
        
    return Error
        
def get_subject_ids(PATH="D://mesa/polysomnography/annotations-events-nsrr"):
    files = os.listdir(PATH)
    subject_ids = []

    for file in files:
        subject_id = str(file.split('-')[2])
        if subject_id not in subject_ids:
            subject_ids.append(subject_id)

    return subject_ids