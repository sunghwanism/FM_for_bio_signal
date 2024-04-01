import os
import random
import shutil
import zipfile

zip_file_path = r"C:\Users\user\Downloads\pair_test.zip"
output_folder = r"C:\Users\user\Downloads\pair_test"

os.makedirs(output_folder, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(output_folder)

subjects = set()
for file_name in os.listdir(output_folder):
    if file_name.endswith('.npz'):
        subject = file_name.split('_')[0]
        subjects.add(subject)

selected_subjects = random.sample(subjects, 10)

def split_files(files):
    random.shuffle(files)
    num_files = len(files)
    train_count = int(num_files * 0.7)
    val_count = int(num_files * 0.15)
    test_count = num_files - train_count - val_count
    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]
    return train_files, val_files, test_files

for subject in selected_subjects:
    subject_files = [file_name for file_name in os.listdir(output_folder) if file_name.startswith(subject)]
    train_files, val_files, test_files = split_files(subject_files)
    
    train_folder = os.path.join(output_folder, f"subj_{subject}_train")
    val_folder = os.path.join(output_folder, f"subj_{subject}_val")
    test_folder = os.path.join(output_folder, f"subj_{subject}_test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    for file_name in train_files:
        shutil.move(os.path.join(output_folder, file_name), os.path.join(train_folder, file_name))
    for file_name in val_files:
        shutil.move(os.path.join(output_folder, file_name), os.path.join(val_folder, file_name))
    for file_name in test_files:
        shutil.move(os.path.join(output_folder, file_name), os.path.join(test_folder, file_name))