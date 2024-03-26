import os
import numpy as np

def process_ecg_data(file_path, block_size):
    ecg_data_blocks = []
    ecg_status_blocks = []
    ecg_label_blocks = []

    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            if line.strip() != '':
                row_data = line.split(',')
                ecg_data_blocks.append(float(row_data[0]))
                ecg_status_blocks.append(int(row_data[-1]))

        label = os.path.basename(file_path).split('_')[1]
        ecg_label_blocks.extend([label] * len(ecg_data_blocks))

    num_blocks = len(ecg_data_blocks) // block_size
    ecg_data_blocks = np.split(np.array(ecg_data_blocks)[:num_blocks * block_size], num_blocks)
    
    return ecg_data_blocks


def main():
    random_seed = 42
    np.random.seed(random_seed)
    
    data_directory = '/NFS/Users/moonsh/data/mesa/preproc/final'
    save_dir = '/NFS/Users/moonsh/data/mesa/preproc/pair'
    
    ecg_data_blocks = []

    for filename in os.listdir(data_directory):
        if filename.endswith('.csv') and 'ecg' in filename:
            file_path = os.path.join(data_directory, filename)
            data_blocks = process_ecg_data(file_path, block_size=7680)
            ecg_data_blocks.extend(data_blocks)

    ecg_data_array = np.array(ecg_data_blocks)
    shuffle_indices = np.random.permutation(len(ecg_data_array))
    ecg_data_array_shuffled = ecg_data_array[shuffle_indices]

    np.save(os.path.join(save_dir, 'ecg_data_shuffled.npy'), ecg_data_array_shuffled)

if __name__ == '__main__':
    main()