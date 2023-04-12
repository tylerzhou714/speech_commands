import tensorflow as tf

model = tf.keras.models.load_model('model/best_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model/my_speech_recognition_model.tflite', 'wb') as f:
    f.write(tflite_model)
