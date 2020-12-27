import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text as text

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import os
import string


def build_classifier_model():
    text_input = tf.keras.layers.Input(
        shape=(), dtype=tf.string, name='inputs')
    preprocessing_layer = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/albert_en_preprocess/2", name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/2",
                             trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(32, activation="relu")(net)
    net = tf.keras.layers.Dense(
        20, activation="softmax", name='classifier')(net)
    return tf.keras.Model(text_input, net)
