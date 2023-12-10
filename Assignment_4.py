'''
Assignment No.04
Name: Chinmay Bitne
Roll No: 14
Batch: B1
Title: "Implement Bi-gram, Tri-gram word sequence and its count in text input
data using NLTK library"
'''

from nltk import ngrams
from nltk.util import ngrams

#unigram model
n = 1
sentence = "The Moon is Earth's only natural satellite. It orbits at an average distance of 384400 km, about 30 times Earth's diameter."

unigrams = ngrams(sentence.split(), n)
print(f"\n++++++++++++++++++   UNIGRAM    ++++++++++++++++++++++++++++")
for item in unigrams:
    print(item)
#bigram model
n = 2
unigrams = ngrams(sentence.split(), n)
print(f"\n++++++++++++++++++   BIGRAM    ++++++++++++++++++++++++++++")
for item in unigrams:
    print(item)

#trigram model
n = 3
unigrams = ngrams(sentence.split(), n)
print(f"\n++++++++++++++++++   TRIGRAM    ++++++++++++++++++++++++++++")
for item in unigrams:
    print(item)

'''
Output:

++++++++++++++++++   UNIGRAM    ++++++++++++++++++++++++++++
('The',)
('Moon',)
('is',)
("Earth's",)
('only',)
('natural',)
('satellite.',)
('It',)
('orbits',)
('at',)
('an',)
('average',)
('distance',)
('of',)
('384400',)
('km,',)
('about',)
('30',)
('times',)
("Earth's",)
('diameter.',)

++++++++++++++++++   BIGRAM    ++++++++++++++++++++++++++++
('The', 'Moon')
('Moon', 'is')
('is', "Earth's")
("Earth's", 'only')
('only', 'natural')
('natural', 'satellite.')
('satellite.', 'It')
('It', 'orbits')
('orbits', 'at')
('at', 'an')
('an', 'average')
('average', 'distance')
('distance', 'of')
('of', '384400')
('384400', 'km,')
('km,', 'about')
('about', '30')
('30', 'times')
('times', "Earth's")
("Earth's", 'diameter.')

++++++++++++++++++   TRIGRAM    ++++++++++++++++++++++++++++
('The', 'Moon', 'is')
('Moon', 'is', "Earth's")
('is', "Earth's", 'only')
("Earth's", 'only', 'natural')
('only', 'natural', 'satellite.')
('natural', 'satellite.', 'It')
('satellite.', 'It', 'orbits')
('It', 'orbits', 'at')
('orbits', 'at', 'an')
('at', 'an', 'average')
('an', 'average', 'distance')
('average', 'distance', 'of')
('distance', 'of', '384400')
('of', '384400', 'km,')
('384400', 'km,', 'about')
('km,', 'about', '30')
('about', '30', 'times')
('30', 'times', "Earth's")
('times', "Earth's", 'diameter.')
'''