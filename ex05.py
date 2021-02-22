# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import movie_reviews
import random

documents = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = all_words.most_common()[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word,_ in word_features:
        features['contain(%s)' % word] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]

print('NaiveBayes')
classifier = nltk.NaiveBayesClassifier.train(train_set) #0.85
print(nltk.classify.accuracy(classifier, test_set))

print('DecisionTree')
classifier = nltk.DecisionTreeClassifier.train(train_set) # 0.7
print(nltk.classify.accuracy(classifier, test_set))
