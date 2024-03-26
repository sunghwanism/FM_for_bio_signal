
import os
import numpy as np
from tqdm import tqdm


def process_hr_data(file_path, block_size):
    hr_data_blocks = []
    hr_status_blocks = []
    hr_label_blocks = []

    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            if line.strip() != '':
                row_data = line.split(',')
                hr_data_blocks.append(float(row_data[1]))
                hr_status_blocks.append(int(row_data[-1]))

        label = os.path.basename(file_path).split('_')[1].split('.')[0]
        hr_label_blocks.extend([label] * len(hr_data_blocks))

    num_blocks = len(hr_data_blocks) // block_size
    hr_data_blocks = np.split(np.array(hr_data_blocks)[:num_blocks * block_size], num_blocks)
    hr_status_blocks = np.split(np.array(hr_status_blocks)[:num_blocks * block_size], num_blocks)
    hr_label_blocks = np.split(np.array(hr_label_blocks)[:num_blocks * block_size], num_blocks)

    return hr_data_blocks, hr_status_blocks, hr_label_blocks



def main():
    random_seed = 42
    np.random.seed(random_seed)
    
    hr_data_blocks = []
    hr_status_blocks = []
    hr_label_blocks = []
    
    data_directory = '/NFS/Users/moonsh/data/mesa/preproc/final'
    save_dir = '/NFS/Users/moonsh/data/mesa/preproc/pair'

    for filename in tqdm(os.listdir(data_directory)):
        if filename.endswith('.csv') and 'ecg' not in filename:
            file_path = os.path.join(data_directory, filename)
            data_blocks, status_blocks, label_blocks = process_hr_data(file_path, block_size=30)
            hr_data_blocks.extend(data_blocks)
            hr_status_blocks.extend(status_blocks)
            hr_label_blocks.extend(label_blocks)

    hr_data_array = np.array(hr_data_blocks)
    hr_status_array = np.array(hr_status_blocks)
    hr_label_array = np.array(hr_label_blocks)

    shuffle_indices = np.random.permutation(len(hr_data_array))
    hr_data_array_shuffled = hr_data_array[shuffle_indices]
    hr_status_array_shuffled = hr_status_array[shuffle_indices]
    hr_label_array_shuffled = hr_label_array[shuffle_indices]

    np.save(os.path.join(save_dir, 'hr_data_shuffled.npy'), hr_data_array_shuffled)
    np.save(os.path.join(save_dir, 'status_shuffled.npy'), hr_status_array_shuffled)
    np.save(os.path.join(save_dir, 'label_shuffled.npy'), hr_label_array_shuffled)
    

if __name__ == '__main__':
    main()
