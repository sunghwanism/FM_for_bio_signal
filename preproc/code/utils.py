import os

import numpy as np
import pandas as pd



def get_unique_subjects(PATH):
    
    subjects = []
    
    file_list = os.listdir(PATH)
    

    for file in file_list:
        if file.endswith('ecg.csv'):
            subjects.append(file.split('_')[1])
            
    return subjects


def load_data(subject, PATH, load_type=["ECG","HertRate","Active", "psg"]):
    
    print(f"Load Data of Subject {subject}")
    
    result = {"subject":subject,
              "ECG":None,
              "HeartRate":None,
              "Active":None,
              "psg":None}
    
    base_file = f"subject_{subject}.csv"
    base_df = pd.read_csv(os.path.join(PATH, base_file))
    
    if "ECG" in load_type:
        file_name = f"subject_{subject}_ecg.csv"
        df = pd.read_csv(os.path.join(PATH, file_name))
        
        ecg = df["ECG"].to_numpy()
        result["ECG"] = -ecg
        print("Finished loading ECG data", result["ECG"].shape)
        
    if "HertRate" in load_type:
        
        heart_rate = base_df["heart_rate"].to_numpy()
        result["HeartRate"] = heart_rate
        print("Finished loading Heart Rate data", result["HeartRate"].shape)
        
    if "Active" in load_type:

        active = base_df["activity_count"].to_numpy()
        result["Active"] = active
        print("Finished loading Active data", result["Active"].shape)
        
    if "psg" in load_type:
        
        psg = base_df["psg_status"].to_numpy()[::30]
        result["psg"] = psg
        print("Finished loading PSG data", result["psg"].shape)
    

    return result
        
        
        
         