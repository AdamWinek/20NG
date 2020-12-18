import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text as text

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import os
import string


def make_dataset(text, catagories, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((text, catagories))
    dataset = dataset.shuffle(buffer_size=len(text))
    dataset = dataset.batch(batch_size)
    return dataset


def list_of_tuples(l1, l2):
    return list(map(lambda x, y: (x, y), l1, l2))


def clean_text(text):

    # define punctuation

    translator = str.maketrans('', '', string.punctuation)

    text = text.translate(translator).replace('\n', " ").replace('\t', " ")
    return text


def process_data_set(label_to_int):
    TESTDIR = "./20news-bydate/20news-bydate-test"
    TRAINDIR = "./20news-bydate/20news-bydate-train"

    # all catagories of newsGroups
    catagories = ['alt.atheism',
                  'comp.graphics',
                  'comp.os.ms-windows.misc',
                  'comp.sys.ibm.pc.hardware',
                  'comp.sys.mac.hardware',
                  'comp.windows.x',
                  'misc.forsale',
                  'rec.autos',
                  'rec.motorcycles',
                  'rec.sport.baseball',
                  'rec.sport.hockey',
                  'sci.crypt',
                  'sci.electronics',
                  'sci.med',
                  'sci.space',
                  'soc.religion.christian',
                  'talk.politics.guns',
                  'talk.politics.mideast',
                  'talk.politics.misc',
                  'talk.religion.misc']

    test = []
    train = []
    test_labels = []
    train_labels = []
    for cat in catagories:
        path = os.path.join(TESTDIR, cat)
        # map each label to an int
        label_map = label_to_int[cat]

        # Load in test data
        for txt_file in os.listdir(path):
            # open every file in a dataset then append to an array with the text and the category
            file = open(os.path.join(path, txt_file), "r",
                        encoding="utf8", errors='ignore')
            fileText = file.read()

            test.append(clean_text(fileText))
            test_labels.append(label_map)
            file.close()

        path = os.path.join(TRAINDIR, cat)
        # load in train data
        for txt_file in os.listdir(path):
            # open every file in a dataset then append to an array with the text and the category
            file = open(os.path.join(path, txt_file), "r",
                        encoding="utf8", errors='ignore')
            fileText = file.read()
            train_labels.append(label_map)
            train.append(clean_text(fileText))
            file.close()

    # create catagorical array and map labels to catagorical array
    train_cats = tf.keras.utils.to_categorical(train_labels, num_classes=20)
    test_cats = tf.keras.utils.to_categorical(test_labels, num_classes=20)

    test_dataset = make_dataset(test, test_cats, 32)
    train_dataset = make_dataset(train, train_cats, 32)
    return(test_dataset, train_dataset)


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


label_mappings = dict({
    'alt.atheism': 0,
    'comp.graphics': 1,
    'comp.os.ms-windows.misc': 2,
    'comp.sys.ibm.pc.hardware': 3,
    'comp.sys.mac.hardware': 4,
    'comp.windows.x': 5,
    'misc.forsale': 6,
    'rec.autos': 7,
    'rec.motorcycles': 8,
    'rec.sport.baseball': 9,
    'rec.sport.hockey': 10,
    'sci.crypt': 11,
    'sci.electronics': 12,
    'sci.med': 13,
    'sci.space': 14,
    'soc.religion.christian': 15,
    'talk.politics.guns': 16,
    'talk.politics.mideast': 17,
    'talk.politics.misc': 18,
    'talk.religion.misc': 19
})


def build_classifier_model_with_bert():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/albert_en_preprocess/2", name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/2",
                             trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(20, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)


# get data returns  (test_dataset, train_dataset)
data = process_data_set(label_mappings)
classifier_model = build_classifier_model_with_bert()
classifier_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                         optimizer=tf.keras.optimizers.Adam(1e-4),
                         metrics=['accuracy'])
history = classifier_model.fit(data[0], epochs=10,
                               validation_data=data[1],
                               validation_steps=30)
print(history)
