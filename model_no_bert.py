import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text as text
from nltk.corpus import stopwords
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
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

    # remove stop words
    STOPWORDS = set(stopwords.words('english'))
    for word in STOPWORDS:
        token = ' ' + word + ' '
        text = text.replace(token, ' ')
        text = text.replace(' ', ' ')

    return text


def createTokenizer(text, vocab_size, oov_token):
    tokenizer = Tokenizer(num_words=vocab_size,
                          oov_token=oov_token, lower=True,)
    tokenizer.fit_on_texts(text)
    word_index = tokenizer.word_index
    print(dict(list(word_index.items())[0:10]))
    print(dict(list(word_index.items())[vocab_size-10:vocab_size]))

    return tokenizer


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
    # test_dataset = make_dataset(test, test_cats, 32)
    # train_dataset = make_dataset(train, train_cats, 32)

    return(test, test_cats, train, train_cats)

# helper function to plot results


def get_average_article_length(articles):
    wordCount = 0
    for article in articles:
        wordCount += len(article.split(" "))
    return int(wordCount / len(articles))


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


tf.get_logger().setLevel('ERROR')

# get data returns  (test_dataset, test_labels,train_dataset, train_labels)
data = process_data_set(label_mappings)


# create tokenizer for text
tokenizer = createTokenizer(data[0], 10000,  "<OOV>")
avg_article_length = get_average_article_length(data[0])
print(avg_article_length)


# create train sequences
train_sequences = tokenizer.texts_to_sequences(data[2])
print(train_sequences[10])
train_padded = pad_sequences(
    train_sequences, maxlen=(avg_article_length + 100), padding="post", truncating="post")

# create test sequences
test_sequences = tokenizer.texts_to_sequences(data[0])
print(train_sequences[10])
test_padded = pad_sequences(
    test_sequences, maxlen=avg_article_length + 100, padding="post", truncating="post")


checkpoint_path = "./training_no_bert_ckpt/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


classifier_model = build_classifier_model(10000)
classifier_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                         optimizer=tf.keras.optimizers.Adam(),
                         metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = classifier_model.fit(np.array(train_padded), data[3], epochs=10,
                               callbacks=[cp_callback], shuffle=True, validation_data=(test_padded, data[1]))


# # print(history)
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
