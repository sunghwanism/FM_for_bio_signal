import numpy as np
import pandas as pd

from source import utils
from source.constants import Constants
from source.preprocessing.epoch import Epoch
from source.preprocessing.heart_rate.heart_rate_service import HeartRateService


class ECGFeatureService(object):
    WINDOW_SIZE = 10 * 30 * 256 - 15 * 256
    # WINDOW_SIZE = 30

    @staticmethod
    def load(subject_id):
        heart_rate_feature_path = ECGFeatureService.get_path(subject_id)
        feature = pd.read_csv(str(heart_rate_feature_path), delimiter=' ').values
        return feature

    @staticmethod
    def get_path(subject_id):
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + '_hr_feature.out')

    @staticmethod
    def write(subject_id, feature):
        heart_rate_feature_path = ECGFeatureService.get_path(subject_id)
        np.savetxt(heart_rate_feature_path, feature, fmt='%f')

    @staticmethod
    def build(subject_id, valid_epochs):
        heart_rate_collection = HeartRateService.load_cropped(subject_id)
        return ECGFeatureService.build_from_collection(heart_rate_collection, valid_epochs)

    @staticmethod
    def build_from_collection(ecg_collection, valid_epochs):
        ecg_features = []

        # interpolated_timestamps, interpolated_ecg = ECGFeatureService.interpolate_and_normalize(
        #     ecg_collection)

        for epoch in valid_epochs:
            indices_in_range = ECGFeatureService.window_epoch(ecg_collection.timestamps, epoch)
            
            ecg_in_range = ecg_collection.values[indices_in_range]
            # feature = HeartRateFeatureService.get_feature(heart_rate_values_in_range)
            
            feature = ecg_in_range
                        
            ecg_features.append(feature)
            
        ecg_features = np.array(ecg_features)

        return ecg_features
       

    @staticmethod
    def get_window(timestamps, epoch):
        start_time = epoch.timestamp - ECGFeatureService.WINDOW_SIZE
        end_time = epoch.timestamp + Epoch.DURATION + ECGFeatureService.WINDOW_SIZE
        timestamps_ravel = timestamps.ravel()
        indices_in_range = np.unravel_index(np.where((timestamps_ravel > start_time) & (timestamps_ravel < end_time)),
                                            timestamps.shape)
        return indices_in_range[0][0]
    
    def window_epoch(timestamps, epoch):
        start_time = epoch.timestamp*256
        end_time = epoch.timestamp*256 + Epoch.DURATION*256
        timestamps_ravel = timestamps.ravel()
        indices_in_range = np.unravel_index(np.where((timestamps_ravel >= start_time) & (timestamps_ravel < end_time)),
                                            timestamps.shape)

        # return indices_in_range[0][0]
        
        return np.arange(start_time, end_time, 1)

    @staticmethod
    def get_feature(heart_rate_values):
        return [np.std(heart_rate_values)]

    @staticmethod
    def interpolate_and_normalize(heart_rate_collection):
        timestamps = heart_rate_collection.timestamps.flatten()
        heart_rate_values = heart_rate_collection.values.flatten()
        interpolated_timestamps = np.arange(np.amin(timestamps),
                                            np.amax(timestamps), 1)
        interpolated_hr = np.interp(interpolated_timestamps, timestamps, heart_rate_values)

        interpolated_hr = utils.convolve_with_dog(interpolated_hr, ECGFeatureService.WINDOW_SIZE)

        scalar = np.percentile(np.abs(interpolated_hr), 90)
        interpolated_hr = interpolated_hr / scalar
        return interpolated_timestamps, interpolated_hr
