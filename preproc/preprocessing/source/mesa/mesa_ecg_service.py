import numpy as np
import pyedflib as pyedflib

from source import utils

#  수정 필요
from source.preprocessing.ecg.ecg_collection import ECGCollection


class MesaECGService(object):
    
    @staticmethod
    def load_raw(file_id):
        project_root = str(utils.get_project_root())

        edf_file = pyedflib.EdfReader(project_root + 'mesa/polysomnography/edfs/mesa-sleep-' + file_id + '.edf')
        
        ecg_col = 0 # ECG column index

        sample_frequencies = edf_file.getSampleFrequencies()

        ecg = edf_file.readSignal(ecg_col)
        sf = sample_frequencies[ecg_col]

        time_hr = np.array(range(0, len(ecg)))  # Get timestamps for heart rate data
        time_hr = time_hr / sf

        data = np.transpose(np.vstack((time_hr, ecg)))
        data = utils.remove_nans(data)
        
        return ECGCollection(subject_id=file_id, data=data)
