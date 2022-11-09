import tensorflow as tf

import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics


def build_and_compile_model():
    model = keras.Sequential()  
    # model.add(Masking(mask_value=0., input_shape=(time_steps, features)))
    # model.add(LSTM(20, return_sequences=True, activation="relu"))
    # model.add(LSTM(1, activation="softmax", return_sequences=False))

    model.add(layers.Dense(42, activation='relu', input_shape=(840,)))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(9, activation='softmax'))
    model.compile(optimizer=optimizers.Adam(0.001),
                  loss=losses.categorical_crossentropy,
                  metrics=[metrics.categorical_accuracy])
    model.summary()
    return model


if __name__ == "__main__":
    build_and_compile_model()
