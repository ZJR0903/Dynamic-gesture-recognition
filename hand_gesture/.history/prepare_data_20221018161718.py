from pathlib import Path
import pandas as pd
import tensorflow as tf

from options import BaseOptions
from serials import serialize_example


def series_to_supervised(data, n_in=1, dropnan=True):
    cols = list()
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
    agg = pd.concat(cols, axis=1)
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def gen_tfrecord(path_str):
    input_feature_path = Path(path_str)

    y_dict = {"standup": [1, 0, 0, 0, 0, 0, 0, 0, 0],
              "sitdown": [0, 1, 0, 0, 0, 0, 0, 0, 0],
              "come": [0, 0, 1, 0, 0, 0, 0, 0, 0],
              "back": [0, 0, 0, 1, 0, 0, 0, 0, 0],
              "stop": [0, 0, 0, 0, 1, 0, 0, 0, 0],
              "left": [0, 0, 0, 0, 0, 1, 0, 0, 0],
              "right": [0, 0, 0, 0, 0, 0, 1, 0, 0],
              "circle": [0, 0, 0, 0, 0, 0, 0, 1, 0],
              "others": [0, 0, 0, 0, 0, 0, 0, 0, 1]}
    time_steps = 20

    for feature_file in input_feature_path.rglob("*/*.txt"):
        # print(feature_file.name)
        # print(video_path.stem)
        # print(video_path.suffix)
        print(feature_file)
        y = [0, 0, 0, 0, 0, 0]
        if "standup" in str(feature_file.parent):
            y = y_dict["standup"]
        if "sitdown" in str(feature_file.parent):
            y = y_dict["sitdown"]
        if "come" in str(feature_file.parent):
            y = y_dict["come"]
        if "back" in str(feature_file.parent):
            y = y_dict["back"]
        if "stop" in str(feature_file.parent):
            y = y_dict["stop"]
        if "left" in str(feature_file.parent):
            y = y_dict["left"]
        if "right" in str(feature_file.parent):
            y = y_dict["right"]
        if "circle" in str(feature_file.parent):
            y = y_dict["circle"]
        if "others" in str(feature_file.parent):
            y = y_dict["others"]

        df = pd.read_table(feature_file, header=None)
        df.drop([42], axis=1, inplace=True)  # 删除最后一列的空列

        df_shift = series_to_supervised(df, time_steps)
        df_flatten = df_shift.values.flatten()
        df_shape = list(df_shift.shape) 
        label_data = y * df_shape[0]
        label_shape = [df_shape[0], len(y)]

        tf_example_serial = serialize_example(df_flatten, df_shape, label_data, label_shape)
        output_file = str(feature_file).replace(".txt", ".tfrecord")
        # if Path(output_file).exists():
        #     print("file exits,skip")
        #     continue
        with tf.io.TFRecordWriter(output_file) as tf_writer:
            tf_writer.write(tf_example_serial)
        print("ok")


if __name__ == '__main__':
    opt = BaseOptions().parse()
    train_set_path = opt.train_path
    test_set_path = opt.test_path
    gen_tfrecord(train_set_path)
    gen_tfrecord(test_set_path)
