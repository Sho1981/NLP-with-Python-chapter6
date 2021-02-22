# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import senseval

def pos_features(context, i):
    if i == 0:
       features = {"prev-word": "<START>",
                   "next-word": context[i+1][1]}
    elif i == len(context):    
       features = {"prev-word": context[i-1][1],
                   "next-word": "<END>"}
    else:
       features = {"prev-word": context[i-1][1],
                   "next-word": context[i+1][1]}
    return features
    
instances = senseval.instances('hard.pos')
features = [(pos_features(instance.context, instance.position), instance.word)
            for instance in instances]
size = int(len(features) * 0.1)
train_set, test_set = features[size:], features[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
