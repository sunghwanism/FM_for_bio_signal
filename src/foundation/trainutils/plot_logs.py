import numpy as np
import matplotlib.pyplot as plt
import zipfile
import io
import os

zip_file_path = r"C:\Users\user\Downloads\logs.zip"

fig, axs = plt.subplots(5, 1, figsize=(12, 12))

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    file_list = zip_ref.namelist()
    npz_files = [file for file in file_list if file.endswith('.npz')]

    for i, npz_file in enumerate(npz_files):
        with zip_ref.open(npz_file) as file:
            with io.BytesIO(file.read()) as f:
                data = np.load(f)
                train_focal_losses, val_focal_losses, train_accuracies, train_advs_losses = data['arr_0']
       
                if i < 1:
                    axs[i].plot(train_focal_losses, label='train_focal_losses')
                    axs[i].plot(val_focal_losses + 10, label='val_focal_losses')
                elif i < 2:
                    axs[i].plot(train_focal_losses, label='train_focal_losses')
                    axs[i].plot(val_focal_losses + 14, label='val_focal_losses')
                elif i < 3:
                    axs[i].plot(train_focal_losses, label='train_focal_losses')
                    axs[i].plot(val_focal_losses + 10, label='val_focal_losses')                                        
                else:
                    axs[i].plot(train_focal_losses, label='train_focal_losses')
                    axs[i].plot(val_focal_losses + 1, label='val_focal_losses')
                
                axs[i].set_title(os.path.basename(npz_file))
                axs[i].legend()

plt.tight_layout()
plt.show()
