import os
import zipfile
import pandas as pd

zip_file_path = r"C:\Users\user\Downloads\pair_test.zip"
temp_dir = "temp_extract"

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

seg_counts = {}

for root, dirs, files in os.walk(os.path.join(temp_dir, "pair_test")):
    for folder in dirs:
        if folder.startswith("subj_"):
            subject_path = os.path.join(root, folder)
            subject_name = folder.split('_')[1]
            npz_files = [file for file in os.listdir(subject_path) if file.endswith('.npz')]
            seg_count = len(npz_files)
            if subject_name in seg_counts:
                seg_counts[subject_name] += seg_count
            else:
                seg_counts[subject_name] = seg_count

counts_df = pd.DataFrame(list(seg_counts.items()), columns=['Subject', 'Num_segment'])
print(counts_df)
