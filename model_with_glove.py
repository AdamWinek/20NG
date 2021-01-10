import numpy as np
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
import os
import urllib
import zipfile


def build_glove_classifier_model(vocab_size, embedding_dimension, embedding_layer):

    model = tf.keras.Sequential([
        embedding_layer,
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),

        #    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(20, activation='softmax')
    ])
    return model


def construct_embedding_layer(vocab_size, embedding_dimension):
    data_directory = './data/glove'

    weights = []
    word2idx = {'PAD': 0}

    with open('{data_directory}/glove.6B.{embedding_dimension}d.txt'.format(data_directory=data_directory, embedding_dimension=embedding_dimension), 'r') as file:
        for index, line in enumerate(file):
            values = line.split()  # Word and weights separated by space
            word = values[0]  # Word is first symbol on each line
            # Remainder of line is weights for word
            word_weights = np.asarray(values[1:], dtype=np.float32)
            # PAD is our zeroth index so shift by one
            word2idx[word] = index + 1
            weights.append(word_weights)

            if index + 1 == vocab_size:
                # Limit vocabulary to top 40k terms
                break

    # Insert the PAD weights at index 0 now we know the embedding dimension
    weights.insert(0, np.random.randn(embedding_dimension))

    # Append unknown and pad to end of vocab and initialize as random
    UNKNOWN_TOKEN = len(weights)
    word2idx['UNK'] = UNKNOWN_TOKEN
    weights.append(np.random.randn(embedding_dimension))

    # Construct our final vocab
    weights = np.asarray(weights, dtype=np.float32)

    glove_weights_initializer = tf.constant(weights)
    # vocab size is vocab size + token for padding + token for oov

    return(tf.keras.layers.Embedding(vocab_size + 2, embedding_dimension, embeddings_initializer=tf.initializers.constant(glove_weights_initializer), trainable=True), word2idx, UNKNOWN_TOKEN)
