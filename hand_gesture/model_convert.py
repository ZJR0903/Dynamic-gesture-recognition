import tensorflow as tf
from pathlib import Path
from options import BaseOptions


# 右手模型转换
def tflite_convert(path_str):
    model_path = Path(path_str)
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_file = model_path.stem + ".tflite"
    open(tflite_file, "wb").write(tflite_model)


if __name__ == '__main__':
    opt = BaseOptions().parse()
    model_normal_path = opt.model_normal_file
    model_reversal_path = opt.model_reversal_file
    tflite_convert(model_normal_path)
    tflite_convert(model_reversal_path)
