import nltk

nltk.download()

import collections
import random
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.metrics import *

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()



music_file =  pd.read_json('files/music_intent_entities.json')
restaurant_file = pd.read_json('files/restaurant_intent_entities.json')
weather_file = pd.read_json('files/weather_intent_entities.json')

data = {}
data['music'] = music_file['text'].to_numpy()
data['restaurant'] = restaurant_file['text'].to_numpy()
data['weather'] = weather_file['text'].to_numpy()

"""Getting the words from the data"""

all_words = []

document = [(text, category) for category in data.keys() for text in data[category]]
random.shuffle(document)

array_words = [nltk.word_tokenize(w) for (w, cat) in document];
flat_list = [word for sent in array_words for word in sent]

"""Removes the **stop words** like ( ‘off’, ‘is’, ‘s’, ‘am’, ‘or’) and  ***non alphabetical*** characters"""

stopWords = set(stopwords.words('english'))


def remove_stop_words(words):
    wordsFiltered = []

    for w in words:
        if w not in stopWords:
            if w.isalpha():
                wordsFiltered.append(w)

    return wordsFiltered


flat_list = remove_stop_words(flat_list)

"""**Lemmatization** i.e., tranforms tarnsforms different forms of words to a single one"""


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatization(words):
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in words]


filtered_list = lemmatization(flat_list)

"""Getting the ***frequency*** of each word and extracting top 2000"""

frequencyDistribution = nltk.FreqDist(w.lower() for w in filtered_list)
word_features = list(frequencyDistribution)[:2000]

"""**FEATURE** **EXTRACTION**"""


def feature_Extraction(doc):
    document_words = [word.lower() for word in nltk.word_tokenize(doc)]

    document_words = remove_stop_words(document_words)
    document_words = lemmatization(document_words)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


"""Training the model"""

test_set = nltk.classify.apply_features(feature_Extraction, document[:500])
train_set = nltk.classify.apply_features(feature_Extraction, document[500:])
classifier = nltk.NaiveBayesClassifier.train(train_set)

"""Testing the model *accuracy*"""

print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(20)

"""Measuring **Precision,Recall,F-Measure** of a classifier.Finding **Confusion matrix**"""

actual_set  = collections.defaultdict(set)
predicted_set  = collections.defaultdict(set)
# cm here refers to confusion matrix
actual_set_cm = []
predicted_set_cm = []

for i, (feature, label) in enumerate(test_set):
    actual_set[label].add(i)
    actual_set_cm.append(label)
    predicted_label = classifier.classify(feature)
    predicted_set[predicted_label].add(i)
    predicted_set_cm.append(predicted_label)

for category in data.keys():
  print(category,'precision :',precision(actual_set[category], predicted_set[category]))
  print(category,'recall :',recall(actual_set[category], predicted_set[category]))
  print(category,'f-measure :',f_measure(actual_set[category], predicted_set[category]))

confusion_matrix = ConfusionMatrix(actual_set_cm, predicted_set_cm)
print(confusion_matrix)

"""**OUTPUTS**"""

print(classifier.classify(feature_Extraction("Is it sunnier today?")))
print(classifier.classify(feature_Extraction("book a table")))
print(classifier.classify(feature_Extraction(" I want to listen to popular telugu song ")))