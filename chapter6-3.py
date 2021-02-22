import nltk
import random
from nltk.corpus import brown

"""
6.3.1
"""
tagged_sents = list(brown.tagged_sents(categories='news'))
random.shuffle(tagged_sents)
size = int(len(tagged_sents) * 0.1)
train_set, test_set = tagged_sents[size:], tagged_sents[:size]

file_ids = brown.fileids(categories='news')
size = int(len(file_ids) * 0.1)
train_set = brown.tagged_sents(file_ids[size:]) 
test_set = brown.tagged_sents(file_ids[:size])

train_set = brown.tagged_sents(categories='news') 
test_set = brown.tagged_sents(categories='fiction')

"""
6.3.2
"""
classifier = nltk.NaiveBayesClassifier.train(train_set)
print('Accuracy: %4.2f' % nltk.classify.accuracy(classifier, test_set))

"""
6.3.3
"""
"""
6.3.4
"""
tagged_sents = brown.tagged_sents(categories='news')
size = int(len(tagged_sents) * 0.9)
train_sents = tagged_sents[:size]
test_sents = tagged_sents[size:]
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)

def tag_list(tagged_sents):
    return [tag for sent in tagged_sents for (word, tag) in sent]

def apply_tagger(tagger, corpus):
    return [tagger.tag(nltk.tag.untag(sent)) for sent in corpus]

gold = tag_list(brown.tagged_sents(categories='editorial'))
test = tag_list(apply_tagger(t2, brown.tagged_sents(categories='editorial')))
cm = nltk.ConfusionMatrix(gold, test)
