# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import brown

def dialogue_act_features(post, history):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains(%s)' % word.lower()] = True
    if len(history) == 0:
        features['prev-class'] = "<START>"
    else:
        features['prev-class'] = history[-1]
    return features

posts = nltk.corpus.nps_chat.xml_posts()[:10000]

history, featuresets = [], []
h_append, f_append = history.append, featuresets.append
for post in posts:
    f_append((dialogue_act_features(post.text, history), post.get('class')))
    h_append(post.get('class'))

size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
print(classifier.show_most_informative_features(30))