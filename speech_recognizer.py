import os
import pickle
import sys
import numpy as np
import sounddevice as sd
import librosa
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit
from keras.models import load_model
from scipy.io.wavfile import write

class SpeechRecognizer(QWidget):
    def __init__(self):
        super().__init__()

        self.model = load_model('model/best_model_all.h5')
        with open('model/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('语音识别器')

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        self.start_button = QPushButton('开始录音')
        self.start_button.clicked.connect(self.start_recording)
        hbox.addWidget(self.start_button)

        vbox.addLayout(hbox)

        self.output_label = QLabel("输出:")
        vbox.addWidget(self.output_label)

        self.output_text = QTextEdit()
        vbox.addWidget(self.output_text)

        self.setLayout(vbox)

    def start_recording(self):
        self.output_text.clear()

        duration = 1  # 3 秒
        fs = 16000
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        recording = np.squeeze(recording)

        # 保存录音到本地 data 目录
        if not os.path.exists('data'):
            os.makedirs('data')
        wav_file_path = os.path.join('data', 'last_recording.wav')
        write(wav_file_path, fs, recording)

        mfcc = librosa.feature.mfcc(y=recording, sr=fs, n_mfcc=64)
        mfcc = mfcc.T
        mfcc = mfcc.reshape(1, 32, 64, 1)

        predictions = self.model.predict(mfcc)
        predicted_index = np.argmax(predictions)
        predicted_label = self.label_encoder.inverse_transform([predicted_index])[0]
        self.output_text.clear()
        self.output_text.append(f"识别的命令: {predicted_label}")

if __name__ == '__main__':
    app = QApplication(sys.argv)

    speech_recognizer = SpeechRecognizer()
    speech_recognizer.show()

    sys.exit(app.exec_())
