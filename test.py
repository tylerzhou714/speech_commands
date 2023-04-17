import librosa
from tensorflow.keras.models import load_model
from data_load import *
import pickle

# 加载模型
model = load_model('model/best_model_all.h5')


# 在预测时，从文件加载 label_encoder
with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


# 对输入语音进行预处理
def preprocess_audio(file_path, n_mfcc=64, n_frames=32):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc = mfcc.T

    # 填充或截断以获得所需的帧数
    if mfcc.shape[0] < n_frames:
        mfcc = np.pad(mfcc, ((0, n_frames - mfcc.shape[0]), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:n_frames, :]

    return mfcc


# 使用模型进行预测
def predict_command(model, audio_data):
    audio_data = audio_data.reshape(1, audio_data.shape[0], audio_data.shape[1], 1)
    predicted_probs = model.predict(audio_data)
    predicted_label = np.argmax(predicted_probs, axis=1)
    command = label_encoder.inverse_transform(predicted_label)[0]
    return command


# 示例
audio_file = 'data/1daa5ada_nohash_0.wav'
audio_data = preprocess_audio(audio_file)
command = predict_command(model, audio_data)

print(f"Predicted command: {command}")
