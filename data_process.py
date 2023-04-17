import os
import librosa
import numpy as np


def preprocess_data(data_dir, output_dir, test_list_path, val_list_path):
    # 读取测试集和验证集文件名
    with open(test_list_path) as f:
        test_files = set(f.read().splitlines())

    with open(val_list_path) as f:
        val_files = set(f.read().splitlines())

    labels = os.listdir(data_dir)
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            files = os.listdir(label_dir)
            for file in files:
                file_path = label_dir + '/'+ file
                if file_path.endswith(".wav"):
                    mfcc = librosa.feature.mfcc(y=librosa.load(file_path, sr=None)[0], n_mfcc=64)
                    mfcc = mfcc.T
                    if file_path.replace(data_dir, "")[0:] in test_files:
                        output_subdir = os.path.join(output_dir, 'test')
                    elif file_path.replace(data_dir, "")[0:] in val_files:
                        output_subdir = os.path.join(output_dir, 'val')
                    else:
                        output_subdir = os.path.join(output_dir, 'train')

                    os.makedirs(output_subdir, exist_ok=True)
                    output_file_path = os.path.join(output_subdir, f"{label}_{file}.npy")
                    np.save(output_file_path, mfcc)


preprocess_data(r"G:/speech_commands_v0.01/", "./preprocessed_data/",
                r"G:/speech_commands_v0.01/testing_list.txt",
                r"G:/speech_commands_v0.01/validation_list.txt")
