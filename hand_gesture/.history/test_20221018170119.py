import tensorflow as tf
from options import BaseOptions
from serials import parse_function
from train import get_dataset


def evaluate(model_path_str, dataset):
    model = tf.keras.models.load_model(model_path_str)
    score = model.evaluate(dataset)
    print("test loss:", score[0])
    print("test scores:", score[1])


if __name__ == '__main__':
    opt = BaseOptions().parse()
    test_path = opt.test_path
    # right_hand
    dataset_normal = get_dataset(test_path, "normal")
    model_normal = opt.model_normal_file
    evaluate(model_normal, dataset_normal)
    # left_hand
    dataset_reversal = get_dataset(test_path, "reversal")
    model_reversal = opt.model_reversal_file
    evaluate(model_reversal, dataset_reversal)

10.20-12.16

3.5.6