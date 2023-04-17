import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from cnn import build_model
from data_load import load_data
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def train():
    data_dir = "./preprocessed_data/"

    # 使用新的 load_data 函数加载数据
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_data(data_dir)

    # 将整数标签 one-hot 编码
    y_train_onehot = to_categorical(y_train)
    y_val_onehot = to_categorical(y_val)
    y_test_onehot = to_categorical(y_test)

    # 构建模型
    input_shape = (32, 64, 1)
    num_classes = 31
    model = build_model(input_shape, num_classes)

    # 调整数据形状以符合模型输入要求
    X_train = X_train.reshape(X_train.shape[0], 32, 64, 1)
    X_val = X_val.reshape(X_val.shape[0], 32, 64, 1)
    X_test = X_test.reshape(X_test.shape[0], 32, 64, 1)

    # 创建回调
    checkpoint = tf.keras.callbacks.ModelCheckpoint('model/my_speech_recognition_model_{epoch:02d}.h5',
                                                    save_freq='epoch',
                                                    save_weights_only=False,
                                                    period=10)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

    # 训练模型
    history = model.fit(X_train, y_train_onehot,
                        epochs=100,
                        validation_data=(X_val, y_val_onehot),
                        callbacks=[checkpoint, early_stopping])

    # 评估模型性能
    test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot)
    print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

    # 保存模型
    # 在训练函数中，保存 label_encoder 到文件
    with open('model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    model.save('model/best_model_all.h5')

    # 绘制散点图
    train_loss = history.history['loss']
    train_accuracy = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_accuracy']

    data = {'loss': train_loss + val_loss + [test_loss],
            'accuracy': train_accuracy + val_accuracy + [test_accuracy],
            'type': ['train'] * len(train_loss) + ['val'] * len(val_loss) + ['test']}

    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='loss', y='accuracy', hue='type')
    plt.title("Scatterplot of Loss vs. Accuracy for Train, Validation, and Test Sets")
    plt.show()


    # 在测试集上进行预测
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # 计算分类报告和混淆矩阵
    report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)
    conf_matrix = confusion_matrix(y_test, y_pred_classes)

    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)


if __name__ == "__main__":
    train()
