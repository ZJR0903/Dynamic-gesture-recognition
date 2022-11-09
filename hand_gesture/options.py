import argparse


class BaseOptions:

    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initializer(self):
        self._parser.add_argument("-tr", "--train_path", default="/home/yang/newDisk/dataset/hand_video/train")
        self._parser.add_argument("-tt", "--test_path", default="/home/yang/newDisk/dataset/hand_video/test")
        self._parser.add_argument("-fn", "--model_normal_file", default="hand_gesture_normal.h5", help="path to output")
        self._parser.add_argument("-fr", "--model_reversal_file", default="hand_gesture_reversal.h5", help="path to output")

        self._parser.add_argument("-o", "--output_path", default="output", help="path to output")
        self._parser.add_argument("-b", "--batch_size", type=int, default=4, help="input batch size")
        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initializer()
        self._opt = self._parser.parse_args()
        return self._opt
