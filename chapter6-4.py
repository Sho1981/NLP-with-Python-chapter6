import nltk
import random
from nltk.corpus import brown
import math

"""
6.4.1
"""
def entropy(labels):
    freqdist = nltk.FreqDist(labels)
    probs = [freqdist.freq(l) for l in freqdist]
    print(probs)
    return -sum(p * math.log(p,2) for p in probs)

labels = ['M', 'F', 'M', 'F']
print(entropy(labels))
