import numpy as np
import matplotlib.pyplot as plt

stage_counts = {}

for root, dirs, files in os.walk(os.path.join(temp_dir, "pair_test")):
    for folder in dirs:
        if folder.startswith("subj_"):
            subject_name = folder.split('_')[1]
            subject_path = os.path.join(root, folder)
            stage_counts[subject_name] = {'0': 0, '1': 0, '2': 0, '3': 0}
            
            for file_name in os.listdir(subject_path):
                if file_name.endswith('.npz'):
                    npz_file_path = os.path.join(subject_path, file_name)
                    try:
                        npz_data = np.load(npz_file_path)
                        stage = npz_data['stage']
                        
                        if stage == 0:
                            stage_counts[subject_name]['0'] += 1
                        elif stage in [1]:
                            stage_counts[subject_name]['1'] += 1
                        elif stage in [2]:
                            stage_counts[subject_name]['2'] += 1
                        elif stage in [3, 4, 5]:
                            stage_counts[subject_name]['3'] += 1
                    except Exception as e:
                        print(f"Error processing file {npz_file_path}: {e}")


stage_counts_df = pd.DataFrame(stage_counts).T

fig, axes = plt.subplots(nrows=1, ncols=len(stage_counts_df), figsize=(15, 5), sharey=True)
for i, (subject, counts) in enumerate(stage_counts_df.iterrows()):
    counts.plot(kind='bar', ax=axes[i], title=f'Subject {subject} Label Distribution')
    axes[i].set_xlabel('Sleep Stage')
    axes[i].set_ylabel('Frequency')


plt.tight_layout()
plt.show()
