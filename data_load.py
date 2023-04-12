import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(data_dir):
    def load_files(subdir):
        data = []
        labels = []
        for file in os.listdir(os.path.join(data_dir, subdir)):
            if file.endswith(".npy"):
                label = file.split("_")[0]
                loaded_data = np.load(os.path.join(data_dir, subdir, file))

                # 如果数据形状不符合要求，您可以选择截断或填充数据以满足 (32, 64) 的形状要求
                if loaded_data.shape != (32, 64):
                    # 如果数据维度大于 (64, 32)，则截断数据
                    if loaded_data.shape[0] > 32:
                        loaded_data = loaded_data[:32, :]
                    # 如果数据维度小于 (64, 32)，则填充数据
                    elif loaded_data.shape[0] < 32:
                        padded_data = np.zeros((32, 64))
                        padded_data[:loaded_data.shape[0], :] = loaded_data
                        loaded_data = padded_data

                data.append(loaded_data)
                labels.append(label)
        return data, labels

    X_train, y_train = load_files('train')
    X_val, y_val = load_files('val')
    X_test, y_test = load_files('test')

    # 将字符串标签编码为整数
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    return np.array(X_train), np.array(X_val), np.array(X_test), np.array(y_train_encoded), np.array(y_val_encoded), np.array(y_test_encoded), label_encoder



