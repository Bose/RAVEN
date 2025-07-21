import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import config

parent_folder = config.MUSAN_FOLDER_PATH
project_folder = config.PROJECT_FOLDER_PATH
dirs = ['music', 'noise']


data = []

for dir_name in dirs:
    dir_path = os.path.join(parent_folder, dir_name)
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.wav'):
                relative_path = os.path.relpath(os.path.join(root, file), parent_folder)
                subfolder = os.path.relpath(root, dir_path)  # Get subfolder relative to the directory
                if dir_name == 'musan':
                    dir_name = 'music'
                data.append({
                    'filepath': relative_path,
                    'type': dir_name
                })

df = pd.DataFrame(data)


musan_df = df[df['type'] == 'music']
musan_train, musan_test = train_test_split(musan_df, test_size=0.2, random_state=42)
musan_val, musan_test = train_test_split(musan_test, test_size=0.5, random_state=42)


noise_df = df[df['type'] == 'noise']
noise_train, noise_test = train_test_split(noise_df, test_size=0.2, random_state=42)
noise_val, noise_test = train_test_split(noise_test, test_size=0.5, random_state=42)


musan_train['split'] = 'train'
musan_val['split'] = 'val'
musan_test['split'] = 'test'

noise_train['split'] = 'train'
noise_val['split'] = 'val'
noise_test['split'] = 'test'

csv_save_path = os.path.join(project_folder, 'data', 'musan_split.csv')
final_df = pd.concat([musan_train, musan_val, musan_test, noise_train, noise_val, noise_test])

final_df.to_csv(csv_save_path, index=False)
