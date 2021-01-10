import matplotlib.pyplot as plt
from data import get_x_percent_length, process_data_set, clean_text, createTokenizer, handle_padding, download_glove_dataset, words_to_glove_idx
from model_with_glove import build_glove_classifier_model, construct_embedding_layer
from model_custom_embeddings import build_classifier_model
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# helper method to plot graphs at the completion of training
# pass in the trained model and the name of the metric you want graphed


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


def run_model(TESTDIR, TRAINDIR, run_name, modelType, embedding_dimension, vocab_size, training_epochs, padding_cutoff):
    # set the tensorflow error settings
    tf.get_logger().setLevel('ERROR')

    # created checkpoint saving information
    checkpoint_path = f"./checkpoints/{run_name}" + "/{epoch:02d}"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # get data returns  (test_dataset, test_labels,train_dataset, train_labels)
    # pass in the location of testing and training data
    dataset = process_data_set(TESTDIR, TRAINDIR)
    (test_data, test_labels, train_data, train_labels) = dataset

    model_with_tokenized_data = None

    # run different models based on the user input
    if (modelType == "glove"):
        model_with_tokenized_data = run_glove(
            embedding_dimension, vocab_size, padding_cutoff, dataset)
    elif (modelType == "custom"):
        model_with_tokenized_data = run_with_custom_embeddings(
            embedding_dimension, vocab_size, padding_cutoff, dataset)
    else:
        raise(RuntimeError("incorrect model information passed in"))

    (compiled_model, vectorized_test, vectorized_train) = model_with_tokenized_data
    history = compiled_model.fit(np.array(vectorized_train), train_labels, epochs=training_epochs,
                                 callbacks=[cp_callback], shuffle=True, validation_data=(vectorized_test, test_labels))

    # prints metrics related to the training
    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")


def run_glove(embedding_dimension, vocab_size,
              padding_cutoff, dataset):

    # exract specific data from dataset tuple
    (test_data, test_labels, train_data, train_labels) = dataset

    # Available dimensions for 6B data is 50, 100, 200, 300
    download_glove_dataset(embedding_dimension)
    # construct glove embeddings from file
    (embedding_layer, word2idx, UNKNOWN_TOKEN) = construct_embedding_layer(
        vocab_size, embedding_dimension)

    # associate each word in the dataset with an int
    converted_test = words_to_glove_idx(train_data, word2idx, UNKNOWN_TOKEN)
    converted_train = words_to_glove_idx(test_data, word2idx, UNKNOWN_TOKEN)

    # pad the data so all examples are of the same length
    max_len = get_x_percent_length(80, train_data)
    padded_test = pad_sequences(
        converted_test, maxlen=(max_len), padding="post", truncating="post")
    padded_train = pad_sequences(
        converted_train, maxlen=(max_len), padding="post", truncating="post")

    # buld the model
    classifier_model = build_glove_classifier_model(
        vocab_size, embedding_dimension, embedding_layer)
    model = classifier_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                     optimizer=tf.keras.optimizers.Adam(),
                                     metrics=[tf.keras.metrics.CategoricalAccuracy()])
    return(model, padded_test, padded_train)


# TESTDIR = "./20news-bydate/20news-bydate-test"
# TRAINDIR = "./20news-bydate/20news-bydate-train"

# embedding_dimension = 100
# use_glove = True
# if(use_glove):


def run_with_custom_embeddings(embedding_dimension, vocab_size, padding_cutoff, dataset):
    # exract specific data from dataset tuple
    (test_data, test_labels, train_data, train_labels) = dataset

    # creates a tokenizer from training data
    tokenizer = createTokenizer(train_data, vocab_size,  "<OOV>")

    # created padded test sequences
    max_len = get_x_percent_length(80, train_data)
    padded_test = handle_padding(max_len, test_data, tokenizer)
    padded_train = handle_padding(max_len, train_data, tokenizer)

    checkpoint_path = "./checkpoints/training_no_bert/{epoch:02d}"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # buld the model
    classifier_model = build_classifier_model(10000, embedding_dimension)
    model = classifier_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                     optimizer=tf.keras.optimizers.Adam(),
                                     metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return(model, padded_test, padded_train)
