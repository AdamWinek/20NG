import numpy as np
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub


def build_classifier_model(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 128),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),

        #    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(20, activation='softmax')
    ])
    return model


# FIXME yicheng: as an organizational note, I would suggest splitting this into
# three files for now:
#
# data.py -- this handles loading the data of 20NG, computing the vocab, and all
#   the input data related functionalities
# model.py -- this handles the actual keras model construction, and exposes just
#   the model
# main.py -- this drives everything together and handles command line input. You
#   might want to factor out the bottom of this file into a "main function" with
#   some inputs that you can specify over the command line. You might want to
#   familiarize yourself with the ArgumentParser class in python to do so.
