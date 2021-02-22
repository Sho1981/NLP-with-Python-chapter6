# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import brown

def features(sent):
   feature = {}
   for word in sent:
      if word not in ['strong', 'powerful']:
         feature["contains(%s)" % word.lower()] = True
   return feature

featuresets = []
for cat in brown.categories():
   brown_sents = brown.sents(categories=cat)
   strong_sents = [(sent, 'strong') for sent in brown_sents if 'strong' in sent]
   powerful_sents = [(sent, 'powerful') for sent in brown_sents if 'powerful' in sent]
   allsents = strong_sents + powerful_sents
   featuresets.extend([(features(sent), pattern) for (sent, pattern) in allsents])

size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
print(classifier.show_most_informative_features(10))