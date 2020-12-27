import matplotlib.pyplot as plt
from data import get_x_percent_length, process_data_set, clean_text, createTokenizer, handle_padding
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
classifier_model = build_classifier_model(10000)
classifier_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                         optimizer=tf.keras.optimizers.Adam(),
                         metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = classifier_model.fit(np.array(padded_train), train_labels, epochs=10,
                               callbacks=[cp_callback], shuffle=True, validation_data=(padded_test, test_labels))


# prints metrics related to the training
plot_graphs(history, "Accuracy")
plot_graphs(history, "Loss")
