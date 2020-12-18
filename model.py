import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text as text

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import os
import string


def make_dataset(text, catagories, batch_size):

    # labels = tf.constant(catagories)
    print(catagories.ndim)
    print(np.array(text).shape)
    dataset = tf.data.Dataset.from_tensor_slices(
        list_of_tuples(tf.constant(np.array(text)), tf.constant(catagories))
    )
    dataset = dataset.shuffle(buffer_size=9000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    print(dataset.element_spec)

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
    # creates arrays of labels and categrories for each NG
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

    # I am not using datasets because they are complicating everything
    #test_dataset = make_dataset(test, test_cats, 32)
    #train_dataset = make_dataset(train, train_cats, 32)

    return(test, test_cats, train, train_cats)

# helper function to plot results


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


# mappings of NG label to integer
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
    net = tf.keras.layers.Dense(20, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)


def bert_test():
    bert_preprocess_model = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/albert_en_preprocess/2")

    text_test = ["""From: bjorndahl@augustana.ab.ca
    Subject: Re: document of .RTF
    Organization: Augustana University College, Camrose, Alberta
    Lines: 10

    In article <1993Mar30.113436.7339@worak.kaist.ac.kr>, tjyu@eve.kaist.ac.kr (Yu TaiJung) writes:
    > Does anybody have document of .RTF file or know where I can get it?
    >
    > Thanks in advance. :)

    I got one from Microsoft tech support.

    --
    Sterling G. Bjorndahl, bjorndahl@Augustana.AB.CA or bjorndahl@camrose.uucp
    Augustana University College, Camrose, Alberta, Canada      (403) 679-1100
    """]
    text_preprocessed = bert_preprocess_model(text_test)
    print(text_preprocessed)
    print(f'Keys       : {list(text_preprocessed.keys())}')
    print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
    print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :100]}')
    print(f'Input Mask : {text_preprocessed["input_mask"][0, :100]}')
    print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :100]}')

    bert_model = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/albert_en_base/2")

    bert_results = bert_model(text_preprocessed)

    print(f'Loaded BERT: {"https://tfhub.dev/tensorflow/albert_en_base/2"}')
    print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
    print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
    print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
    print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')

    classifier_model = build_classifier_model_with_bert()
    bert_raw_result = classifier_model(tf.constant(text_test))
    print(tf.sigmoid(bert_raw_result))


tf.get_logger().setLevel('ERROR')

# get data returns  (test_dataset, train_dataset)
data = process_data_set(label_mappings)
# bert_test()


checkpoint_path = "./training1ckpt/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


classifier_model = build_classifier_model_with_bert()
classifier_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                         optimizer=tf.keras.optimizers.Adam(3e-5),
                         metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = classifier_model.fit(np.array(data[0]), data[1], epochs=10,
                               callbacks=[cp_callback], shuffle=True, validation_split=0.1)


# print(history)
plot_graphs(history, tf.keras.metrics.CategoricalAccuracy())
