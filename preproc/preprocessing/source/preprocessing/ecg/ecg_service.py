import numpy as np
import pandas as pd

from source import utils
from source.constants import Constants
from source.preprocessing.ecg.ecg_collection import ECGCollection


class ECGService(object):

    @staticmethod
    def crop(ecg_collection, interval):
        subject_id = ecg_collection.subject_id
        timestamps = ecg_collection.timestamps
        valid_indices = ((timestamps >= interval.start_time)
                         & (timestamps < interval.end_time)).nonzero()[0]
        
        cropped_data = ecg_collection.data[valid_indices, :]
        
        return ECGCollection(subject_id=subject_id, data=cropped_data)