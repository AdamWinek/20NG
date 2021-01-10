import matplotlib.pyplot as plt
from data import get_x_percent_length, process_data_set, clean_text, createTokenizer, handle_padding, download_glove_dataset, words_to_glove_idx
from model_with_glove import build_glove_classifier_model, construct_embedding_layer
from model_custom_embeddings import build_classifier_model
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import enquiries
from datetime import date

import argparse


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
    elif (modelType == "custom_embeddings"):
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
    converted_test = words_to_glove_idx(test_data, word2idx, UNKNOWN_TOKEN)
    converted_train = words_to_glove_idx(train_data, word2idx, UNKNOWN_TOKEN)

    # pad the data so all examples are of the same length
    max_len = get_x_percent_length(padding_cutoff, train_data)
    padded_test = pad_sequences(
        converted_test, maxlen=(max_len), padding="post", truncating="post")
    padded_train = pad_sequences(
        converted_train, maxlen=(max_len), padding="post", truncating="post")

    # buld the model
    classifier_model = build_glove_classifier_model(
        vocab_size, embedding_dimension, embedding_layer)
    classifier_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                             optimizer=tf.keras.optimizers.Adam(),
                             metrics=[tf.keras.metrics.CategoricalAccuracy()])
    return(classifier_model, padded_test, padded_train)


def run_with_custom_embeddings(embedding_dimension, vocab_size, padding_cutoff, dataset):
    # exract specific data from dataset tuple
    (test_data, test_labels, train_data, train_labels) = dataset

    # creates a tokenizer from training data
    tokenizer = createTokenizer(train_data, vocab_size,  "<OOV>")

    # created padded test sequences
    max_len = get_x_percent_length(padding_cutoff, train_data)
    padded_test = handle_padding(max_len, test_data, tokenizer)
    padded_train = handle_padding(max_len, train_data, tokenizer)

    checkpoint_path = "./checkpoints/training_no_bert/{epoch:02d}"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # buld the model
    classifier_model = build_classifier_model(vocab_size, embedding_dimension)
    classifier_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                             optimizer=tf.keras.optimizers.Adam(),
                             metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return(classifier_model, padded_test, padded_train)


def collect_input_from_user():
    # gets the current date
    today = date.today()
    correctedForm = today.strftime("%b-%d-%Y")

    parser = argparse.ArgumentParser()
    parser.add_argument("-tp", "--test_path",
                        help="Path to the 20NG test dataset", default="./20news-bydate/20news-bydate-test")
    parser.add_argument("-trp", "--train_path",
                        help="Path to the 20NG train dataset", default="./20news-bydate/20news-bydate-train")
    parser.add_argument("-vs", "--vocab_size", type=int,
                        help="Vocab size to be used in the model", default=100000)
    parser.add_argument("-ed", "--embedding_dimension", type=int,
                        help="Number of dimensions used in the vector space to represent words. Must be 50, 100, 200, 300 while using glove", default=100)
    parser.add_argument("-e", "--epochs", type=int,
                        help="number of epochs to be trained on", default=10)
    parser.add_argument(
        "-rn", "--run_name", help="name that will be saved for this training run", default=correctedForm)

    parser.add_argument(
        "-pc", "--padding_cutoff", type=int, help="Percentage of your dataset that will not be cut off by padding",
        default=80)
    args = parser.parse_args()

    model_type = enquiries.choose('Pick a model to run', [
        "glove", "custom_embeddings"])

    # run checks to make sure arguements are valid

    if(args.vocab_size < 0):
        raise ValueError("Vocab Size cannot be negative")

    if(args.vocab_size < 0):
        raise ValueError("Vocab Size cannot be negative")

    if(args.embedding_dimension < 0):
        raise ValueError("Embedding dimensions cannot be negative")

    if(args.epochs < 0 or args.epochs > 50):
        raise ValueError("Number of Epochs must be between 0 and 50")

    if(args.padding_cutoff < 0 or args.padding_cutoff > 100):
        raise ValueError("Percentage to not cutoff must be between 0 and 100")

    if (model_type == "glove" and args.embedding_dimension not in [50, 100, 200, 300]):
        return ValueError("Glove embedding dimension must be 50, 100, 200, or 300 ")

    run_model(args.test_path, args.train_path, args.run_name, model_type,
              args.embedding_dimension, args.vocab_size, args.epochs, args.padding_cutoff)


collect_input_from_user()
