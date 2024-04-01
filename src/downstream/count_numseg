import os
import zipfile
import pandas as pd

zip_file_path = r"C:\Users\user\Downloads\pair_test.zip"

temp_dir = "temp_extract"

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(temp_dir
    )

seg_counts = {}

for subject_folder in os.listdir(os.path.join(temp_dir, "pair_test")):
    subject_name = subject_folder.split('_')[1]
    
    seg_count = 0
    
    subject_path = os.path.join(temp_dir, "pair_test", subject_folder)
    
    seg_count += len([file for file in os.listdir(subject_path) if file.endswith('.npz')])
    
    seg_counts[subject_name] = seg_count

counts_df = pd.DataFrame(list(seg_counts.items()), columns=['Subject', 'Num_segment'])
print(counts_df)
