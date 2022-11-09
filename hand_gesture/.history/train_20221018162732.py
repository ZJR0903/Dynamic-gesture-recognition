import tensorflow as tf
from pathlib import Path
from options import BaseOptions
from keras.callbacks import TensorBoard, ModelCheckpoint

import models
from serials import parse_function


def get_dataset(path_str, dataset_select):
    if dataset_select == "normal":
        tfrecord_path = tf.io.gfile.glob(path_str + "/*/*_normal.tfrecord")
    if dataset_select == "reversal":
        tfrecord_path = tf.io.gfile.glob(path_str + "/*/*_reversal.tfrecord")

    epochs = 100
    buffer_size = 10000
    batch_size = 20
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(lambda x: parse_function(x)).unbatch()
    # print(len(list(parsed_dataset.as_numpy_iterator())))
    dataset = parsed_dataset.repeat(epochs).shuffle(buffer_size).batch(batch_size)
    return dataset


def train(dataset, save_path_str):
    if Path(save_path_str).exists():
        print("model restored!")
        model = tf.keras.models.load_model(save_path_str)
    else:
        model = models.build_and_compile_model()
        
    tensorboard = TensorBoard(log_dir="log")
    check_pointer = ModelCheckpoint(filepath=save_path_str, verbose=1, monitor='val_loss', save_weights_only=False,
                                    save_best_only=True, mode='min', period=1)
    callbacks_list = [tensorboard, check_pointer]
    model.fit(dataset, epochs=10, validation_data=dataset.take(5000), callbacks=callbacks_list)
    # model.fit(dataset, epochs=5, callbacks=callbacks_list)
    # model.save(save_path_str)


if __name__ == '__main__':
    opt = BaseOptions().parse()
    train_path = opt.train_path
    normal_dataset = get_dataset(train_path, "normal")
    reversal_dataset = get_dataset(train_path, "reversal")

    # samples = normal_dataset.take(10)
    # for i in samples:
    #     print(i[1].numpy)
    # print(len(list(normal_dataset.as_numpy_iterator())))
    # print(len(list(reversal_dataset.as_numpy_iterator())))

    train(normal_dataset, opt.model_normal_file)
    train(reversal_dataset, opt.model_reversal_file)
