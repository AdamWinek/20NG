import matplotlib.pyplot as plt
from data import get_x_percent_length, process_data_set, clean_text, createTokenizer, handle_padding, download_glove_dataset
from model_with_glove import build_glove_classifier_model, construct_embedding_layer
from model_custom_embeddings import build_classifier_model
import tensorflow as tf
import numpy as np
# helper method to plot graphs at the completion of training
# pass in the trained model and the name of the metric you want graphed


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


# set the tensorflow error settings
tf.get_logger().setLevel('ERROR')

TESTDIR = "./20news-bydate/20news-bydate-test"
TRAINDIR = "./20news-bydate/20news-bydate-train"
# get data returns  (test_dataset, test_labels,train_dataset, train_labels)
# pass in the location of testing and training data
(test_data, test_labels, train_data,
 train_labels) = process_data_set(TESTDIR, TRAINDIR)
embedding_dimension = 50
use_glove = True
if(use_glove):
    # Available dimensions for 6B data is 50, 100, 200, 300
    download_glove_dataset(embedding_dimension)

    checkpoint_path = "./checkpoints/training_glove/"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    (embedding_layer, word2idx, UNKNOWN_TOKEN) = construct_embedding_layer(
        10000, embedding_dimension, train_labels)

    # associate each word in the dataset with an int

    converted_train = [word2idx.get(
        word, UNKNOWN_TOKEN) for element in train_data for word in element.split()]
    converted_test = [word2idx.get(
        word, UNKNOWN_TOKEN) for element in test_data for word in element.split()]

    # buld the model
    classifier_model = build_glove_classifier_model(
        10000, embedding_dimension, train_data)
    classifier_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                             optimizer=tf.keras.optimizers.Adam(),
                             metrics=[tf.keras.metrics.CategoricalAccuracy()])
    history = classifier_model.fit(np.array(converted_train), train_labels, epochs=10,
                                   callbacks=[cp_callback], shuffle=True, validation_data=(converted_test, test_labels))

    # prints metrics related to the training
    #plot_graphs(history, "Accuracy")
    #plot_graphs(history, "Loss")


else:

    # creates a tokenizer from training data
    tokenizer = createTokenizer(train_data, 10000,  "<OOV>")

    # created padded test sequences
    max_len = get_x_percent_length(80, train_data)
    padded_test = handle_padding(max_len, test_data, tokenizer)
    padded_train = handle_padding(max_len, train_data, tokenizer)

    checkpoint_path = "./checkpoints/training_no_bert/"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # buld the model
    classifier_model = build_classifier_model(10000, embedding_dimension)
    classifier_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                             optimizer=tf.keras.optimizers.Adam(),
                             metrics=[tf.keras.metrics.CategoricalAccuracy()])
    history = classifier_model.fit(np.array(padded_train), train_labels, epochs=10,
                                   callbacks=[cp_callback], shuffle=True, validation_data=(padded_test, test_labels))

    # prints metrics related to the training
    plot_graphs(history, "Accuracy")
    plot_graphs(history, "Loss")
