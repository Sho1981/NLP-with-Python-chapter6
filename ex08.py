# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import movie_reviews
import random
from nltk.corpus import wordnet as wn

documents = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = all_words.most_common()[:2000]

def flatten(nested_list):
    return [e for inner_list in nested_list for e in inner_list]

def synonym(word):
    return flatten(synset.lemma_names() for synset in wn.synsets(word))

def hypernym(word):
    return flatten([syn.lemma_names() for synset in wn.synsets(word)
                                      for syn in synset.hypernyms()])

def document_features(document):
    document_words = set(document)
    for word in set(document):
        if word not in word_features:
            document_words.union(set(synonym(word) + hypernym(word)))
    features = {}
    for word,_ in word_features:
        features['contain(%s)' % word] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
