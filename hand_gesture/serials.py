import tensorflow as tf


def serialize_example(landmarks_data, landmarks_shape,label_data,label_shape):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    feature = {
        'landmarks_data': tf.train.Feature(float_list=tf.train.FloatList(value=landmarks_data)),
        'landmarks_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=landmarks_shape)),
        'label_data': tf.train.Feature(float_list=tf.train.FloatList(value=label_data)),
        'label_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=label_shape)),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def parse_function(example_proto):
    feature_description = {
        'landmarks_data': tf.io.VarLenFeature(dtype=tf.float32),
        'landmarks_shape': tf.io.FixedLenFeature(shape=[2], dtype=tf.int64),
        'label_data': tf.io.VarLenFeature(dtype=tf.float32),
        'label_shape': tf.io.FixedLenFeature(shape=[2], dtype=tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    parsed_example['landmarks_data'] = tf.sparse.to_dense(parsed_example['landmarks_data'])
    parsed_example['label_data'] = tf.sparse.to_dense(parsed_example['label_data'])

    # 还原维度
    landmarks_shape = parsed_example['landmarks_shape']
    label_shape = parsed_example['label_shape']

    landmarks_data = tf.reshape(parsed_example['landmarks_data'], shape=[landmarks_shape[0], -1])
    label_data = tf.reshape(parsed_example['label_data'], shape=[label_shape[0], -1])

    # # 数据归一化
    # mean, variance = tf.nn.moments(mfcc_data, axes=[0, 1])
    # x = tf.nn.batch_normalization(mfcc_data, mean, variance, 0, 1, 0.001)
    # y = landmarks_data / 224.0
    return landmarks_data, label_data