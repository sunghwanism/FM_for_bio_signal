import os

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from collections import Counter

def read_subject_data(subject_id):
    motion_df = pd.read_csv('../data/watch/processed/cropped/{}_cleaned_motion.out'.format(subject_id), sep=' ', names=["second", "x_move", "y_move", "z_move"])
    hr_df = pd.read_csv('../data/watch/processed/cropped/{}_cleaned_hr.out'.format(subject_id), sep=' ', names=["second", "heart_rate"])
    psg_df = pd.read_csv('../data/watch/processed/cropped/{}_cleaned_psg.out'.format(subject_id), sep=' ', names=["second", "psg_status"])
    steps_df = pd.read_csv('../data/watch/processed/cropped/{}_cleaned_counts.out'.format(subject_id), names=["second", "step_count"])

    # clean the psg data to remove '-1' values - turn values with '-1' to 'nan'
    psg_df.replace(-1, float('nan'), inplace=True)

    return (motion_df, hr_df, psg_df, steps_df)


def analyse_sensor_data(sensor_data_df):
    print("   - Starts at", round(min(sensor_data_df.second), 4), "seconds")
    print("   - Ends at", round(max(sensor_data_df.second), 4), "seconds")
    print("\n   - Collects data every", round(max(sensor_data_df.second)/len(sensor_data_df), 4), "seconds")
    
    
def join_sensor_dataframes(motion_df, hr_df, psg_df, steps_df):
  
    # join the dataframes together
    motion_nd_hr_df = pd.merge(motion_df, hr_df, on="second", how="outer")
    motion_hr_nd_steps_df = pd.merge(motion_nd_hr_df, steps_df, on="second", how="outer")
    all_sensor_df = pd.merge(motion_hr_nd_steps_df, psg_df, on="second", how="outer")

    # sort df by second column
    all_sensor_df["second"] = all_sensor_df["second"].astype("float")
    all_sensor_df = all_sensor_df.sort_values("second")
    all_sensor_df = all_sensor_df.reset_index(drop=True)

    return all_sensor_df

def where_no_sensor_value_has_been_recorded_up_till_that_second_turn_this_to_a_val(df, val):

    fixed_df = df.copy()
    # iterate through these columns and fix the 'Nan' values
    for col in fixed_df.drop(columns=["second"], axis=1):

        # iterate backwards over rows and turn "Nan" boxes to the specified val and find where the "Nan" values stop
        i = len(fixed_df[col]) - 1
        while np.isnan(fixed_df[col][i]):
            fixed_df[col][i] = val
            i = i - 1

        # iterate forwards over rows and turn "Nan" boxes to the specified val and find where real values start
        j = 0
        while np.isnan(fixed_df[col][j]):
            fixed_df[col][j] = val
            j = j + 1

    return fixed_df


def fill_in_the_dfs_nan_values(df):

    # fill the "Nan" values with the value encountered before the "Nan" up until it hits the point where the rest of that column is only "Nan" values
    filled_df = df.fillna(method="ffill")

    # fill the "Nan" values with an interpolated value using the value before this "Nan" value and after this "Nan" value
    # filled_df = df.interpolate()

    return filled_df


def get_list_of_averages_for_each_col_of_all_rows(rows, val_to_fill_nans, cols_no_second_list):

    # if there are no values between the specified seconds
    if len(rows) == 0:
        # add empty row
        list_of_averages = list(np.repeat(float("nan"), len(cols_no_second_list)))

    # if there is at least one row with a value between these seconds
    elif len(rows) > 0:
        # average the values in these rows to get just 1 row with the values for each column
        list_of_averages = []
        for col in cols_no_second_list:

            # make sure that the sleep labels column only contains whole numbers - they must maintain their classification label
            if col == 'psg_status':
                # count the number of occurances of each value and take the one that occurs most as this value
                avg = float(rows.loc[:, col].mode()[0])
                    
            # first check if the values are all valid values - replace "Nan" with a value so these are invalid
            elif set(rows.loc[:, col]) != {val_to_fill_nans} and val_to_fill_nans in list(rows.loc[:, col]):
                count_nans = 0
                non_nan_vals = []

                # iterate through these row values to count how many invalid(-10) values there are
                for v in rows.loc[:, col]:
                    if v == val_to_fill_nans:
                        count_nans += 1
                      
                    else:
                        count_nans -= 1
                        non_nan_vals.append(v)

                # if there are more invalid rows than there are proper ones
                if count_nans > 0:
                    avg = sum(non_nan_vals)/len(non_nan_vals)

                # if there are more proper values than invalid ones
                elif count_nans < 0:
                    avg = val_to_fill_nans

                # if there is the exact same amout of invalid values and proper values, use the most common single value or the first of these most common values
                else:
                    print(rows)
                    avg = float(rows.loc[:, col].mode()[0])

            # if all the values are proper values, then just get the means of these values and populate the row with this
            else:
                avg = rows.loc[:, col].mean()
                
            list_of_averages.append(avg)

    return list_of_averages


def turn_seconds_to_a_set_interval(filled_sensor_df, second_column_step, val_to_fill_nans):

    # define a variable for the second column
    sec_col = filled_sensor_df.second

    # get a ist of the cols exluding the second column
    cols_no_second_list = filled_sensor_df.drop(columns=["second"], axis=1).columns

    # get the value of the maximum and minimum second in this dataframe
    min_second_in_df = int(round(min(sec_col) - 0.5))
    max_second_in_df = int(round(max(sec_col) + 0.5))

    # create a new dataframe that we will populate
    new_df = pd.DataFrame(columns=(filled_sensor_df.columns))

    # iterate through each second interval in this dataframe
    for i in np.arange(min_second_in_df, max_second_in_df + second_column_step, second_column_step):

        # get the rows between second "i - 1" and second "i"
        rows = filled_sensor_df.loc[(sec_col > i - (second_column_step - 0.00000001)) & (sec_col < i + 0.00000001)]
        rows = rows.reset_index(drop=True)
        
        list_of_averages = get_list_of_averages_for_each_col_of_all_rows(rows, val_to_fill_nans, cols_no_second_list)

        # create a dataframe row from this list
        new_df_row = pd.DataFrame([i] + list_of_averages, columns=[i], index=filled_sensor_df.columns).T

        # add this row to the new DataFrame
        new_df = pd.concat([new_df, new_df_row], axis=0)

    return new_df

def is_not_nan(v):
  return v == v


def generate_data_to_dic(PATH, second_column_step=1, val_to_fill_nans=-100, print_analysis=False):
    
    subject_ids = []
    for n, filename in enumerate(os.listdir(PATH)):
        filename = filename.split('_')
        subject_id = int(filename[0])
    
        if subject_id not in subject_ids:
            subject_ids.append(subject_id)
        
    # sort the list
    sorted_subject_ids = sorted(subject_ids)

    print("There are", len(sorted_subject_ids), "subject id's\nThey are:\n", sorted_subject_ids)
    
    map_subject_id_to_sensor_dataframes = {}
    
    progress = 0

    for _subject_id in sorted_subject_ids:
        
        # read in the data
        motion_df, hr_df, psg_df, steps_df = read_subject_data(_subject_id)

        if print_analysis:
            # print which subject_id we are analysing
            print("\n--------------------------------------------------------------")
            print("==========================", _subject_id, "==========================")
            print("--------------------------------------------------------------")

            # analyse the data using print statements
            print("The motion sensor:")
            analyse_sensor_data(motion_df)
            print("\nThe heart rate sensor:")
            analyse_sensor_data(hr_df)
            print("\nThe PSG study data:")
            analyse_sensor_data(psg_df)
            print("\nThe step count sensor:")
            analyse_sensor_data(steps_df)

        print(f"{progress+1} Subject...", end="\t")
        progress += 1
        
        # join these sensor dataframes together
        all_sensor_df = join_sensor_dataframes(motion_df, hr_df, psg_df, steps_df)

        # add this subject to the map - map it to its sensor dataframe
        map_subject_id_to_sensor_dataframes[_subject_id] = all_sensor_df            
    
    map_subject_id_to_joined_and_uniform_second_sensor_df = {}
    
    for subject_id, all_sensors_df in map_subject_id_to_sensor_dataframes.items():

        # print the subject_id
        print("\t\t", subject_id)

        # set values where there was no sensor value taken to be a value instead of 'Nan'
        half_filled_df = where_no_sensor_value_has_been_recorded_up_till_that_second_turn_this_to_a_val(all_sensors_df, val_to_fill_nans)

        # fill the Nan values in the dataframe
        filled_sensor_df = fill_in_the_dfs_nan_values(half_filled_df)

        # replace the "-10" values with "Nan" values
        df_val_to_fill_nan_replaced = filled_sensor_df.replace(val_to_fill_nans, float("nan"))

        # drop the rows that have an 'Nan' value for any of the sensors - this means that the sensor hadn't recorded it's first value yet
        no_nan_values = df_val_to_fill_nan_replaced.dropna()

        # turn the second column to a uniform interval by averaging the other columns
        uniform_second_interval_df = turn_seconds_to_a_set_interval(no_nan_values, second_column_step, val_to_fill_nans)

        # fill the nan values in the dataframe
        df_with_one_sec_per_row = fill_in_the_dfs_nan_values(uniform_second_interval_df)

        # reset the index
        df_with_one_sec_per_row.reset_index(drop=True, inplace=True)

        # clean out the rows where there is an invalid sleep state value
        remove_invalid_sleep_state = df_with_one_sec_per_row[is_not_nan(df_with_one_sec_per_row["psg_status"])]
        remove_invalid_sleep_state.reset_index(drop=True, inplace=True)

        # create a new map of the subject_id to this new dataframe
        map_subject_id_to_joined_and_uniform_second_sensor_df[subject_id] = remove_invalid_sleep_state
    
    return map_subject_id_to_joined_and_uniform_second_sensor_df



def filter_to_epoch(csvPATH, bin_size):
    
    subject_ids = []
    for n, filename in enumerate(os.listdir(csvPATH)):
        filename = filename.split('_')
        filename = filename[1].split('.')
        subject_id = int(filename[0])
    
        if subject_id not in subject_ids:
            subject_ids.append(subject_id)
        
    # sort the list
    sorted_subject_ids = sorted(subject_ids)
    
    map_subject_to_df_with_id = {}
    for subject_id in sorted_subject_ids:
        
        fixed_sensor_df = pd.read_csv(csvPATH + f"subject_{subject_id}.csv")
        print("---------------", subject_id, "-----------------")

        print(fixed_sensor_df.shape)
        
        # dropna's
        no_nans_fixed_sensor_df = fixed_sensor_df.dropna()
        print(no_nans_fixed_sensor_df.shape)

        # get the value of the maximum second in this dataframe
        max_second_in_df = int(round(max(no_nans_fixed_sensor_df.second) + 0.5))

        # create a new dataframe that we will populate
        new_df = pd.DataFrame(columns=(list(no_nans_fixed_sensor_df.columns).extend(["session_id"])))

        session_number = 0
        # iterate through each second interval in this dataframe
        for i in np.arange(0, max_second_in_df + bin_size, bin_size):

            # get the rows between second "i - 1" and second "i"
            rows_in_session_df = pd.DataFrame(no_nans_fixed_sensor_df.loc[(no_nans_fixed_sensor_df.second >= (i)) & (no_nans_fixed_sensor_df.second < i + bin_size)])
            
            if not rows_in_session_df.empty:
                # assign the session_id label to this row
                rows_in_session_df['session_id'] = session_number

                # join these rows to the rest of the rows
                new_df = pd.concat([new_df, rows_in_session_df], axis=0)

                session_number += 1

        map_subject_to_df_with_id[subject_id] = new_df
        
    map_subject_id_to_a_map_of_the_session_id_to_psg_status = {}

    for subject_id, sensor_df in map_subject_to_df_with_id.items():

        # for this subject, create a dictionary to map their sessions to their psg status'
        subjects_session_to_psg_map = {}

        for session_id in list(set(sensor_df.session_id)):
        
            # get all id entries in df where psg_status = sleep_state
            all_psg_status = sensor_df[sensor_df['session_id'] == session_id]['psg_status']

            # get the most common psg_status across all rows with this session_id
            most_common_psg_status = Counter(all_psg_status).most_common(1)[0][0]

            # create an entry in the subject dictionary of a map between the session_id and the most common psg_status
            subjects_session_to_psg_map[session_id] = most_common_psg_status

        # add this subjects dictionaries to the map of each subject_id to their dictionaries
        map_subject_id_to_a_map_of_the_session_id_to_psg_status[subject_id] = subjects_session_to_psg_map  
    
    
    return map_subject_to_df_with_id, map_subject_id_to_a_map_of_the_session_id_to_psg_status