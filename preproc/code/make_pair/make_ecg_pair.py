import os
import numpy as np
from tqdm import tqdm

def make_pair_ecg(data_dir, save_dir, data_type='ecg', num_subject=100):
    random_seed = 42
    np.random.seed(random_seed)

    data_list = os.listdir(data_dir)
    data_list = [file for file in data_list if data_type in file]
    data_list.sort()
    data_list = data_list[:num_subject]
    count = 0
    subject_arr = []
    for file in tqdm(data_list):
        subject_id = str(file.split('_')[1])
        subject_arr.append(subject_id)
        if count == 0:
            data_arr = np.load(os.path.join(data_dir, file))
            count+=1
        else:
            data = np.load(os.path.join(data_dir, file))
            data_arr = np.concatenate([data_arr, data], axis=0)

    del data

    return data_arr, subject_arr


def main():
    random_seed = 42
    np.random.seed(random_seed)
    
    data_dir = '/NFS/Users/moonsh/data/mesa/preproc/npy'
    save_dir = '/NFS/Users/moonsh/data/mesa/preproc/pair'
    


    data_arr, subject_arr = make_pair_ecg(data_dir, save_dir)

    shuffle_indices = np.random.permutation(len(ecg_data_array))
    ecg_data_array_shuffled = ecg_data_array[shuffle_indices]

    np.save(os.path.join(save_dir, 'ecg_data_shuffled.npy'), ecg_data_array_shuffled)

if __name__ == '__main__':
    main()