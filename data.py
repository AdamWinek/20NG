from tensorflow.keras.preprocessing.text import Tokenizer
import os
import string
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_x_percent_length(x, articles):
    articleLengths = []

    # get a tuple pair of (wordCount, article)
    for article in articles:
        articleLengths.append((len(article.split(" ")), article))

    # sort by article word count
    sorted_lengths = sorted(articleLengths)
    # return length of the article with x percent length
    return sorted_lengths[int((x * len(articles)) / 100)][0]


def process_data_set(TESTDIR, TRAINDIR):

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
        label_map = label_mappings[cat]

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

    return(test, test_cats, train, train_cats)


# I don't know if this is entirely necessary since I am using a predefined tokenizer but I am leaving it in for now
def clean_text(text):

    # define punctuation
    translator = str.maketrans('', '', string.punctuation)
    # remove punctuation and other troublesome characters that appear in the document
    text = text.translate(translator).replace('\n', " ").replace('\t', " ")
    # remove stop words
    STOPWORDS = set(stopwords.words('english'))
    for word in STOPWORDS:
        token = ' ' + word + ' '
        text = text.replace(token, ' ')
        text = text.replace(' ', ' ')

    return text


def createTokenizer(text, vocab_size, oov_token):
    # creates a tokenizer expects an array of strings as text
    tokenizer = Tokenizer(num_words=vocab_size,
                          oov_token=oov_token, lower=True,)
    tokenizer.fit_on_texts(text)
    word_index = tokenizer.word_index
    print(dict(list(word_index.items())[0:10]))
    print(dict(list(word_index.items())[vocab_size-10:vocab_size]))

    return tokenizer


def handle_padding(max_len, text, tokenizer):
    sequences = tokenizer.texts_to_sequences(text)

    print(sequences[10])
    padded = pad_sequences(
        sequences, maxlen=(max_len), padding="post", truncating="post")
    return padded
