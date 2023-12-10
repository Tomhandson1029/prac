'''
Assignment No.06
Name: Chinmay Bitne
Roll No: 14
Batch: B1
Title: "Implement and visualize Dependency Parsing of Textual Input
using Stan- ford CoreNLP and Spacy library"
'''

import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

text = "The Moon is Earth's only natural satellite. It orbits at an average distance of 384400 km, about 30 times Earth's diameter."

doc = nlp(text)

for token in doc:
    print(
        f"""
TOKEN: {token.text}
=====
{token.tag_ = }
{token.head.text = }
{token.dep_ = }"""
    )

displacy.serve(doc, style="dep")

'''
Output:

TOKEN: The
=====
token.tag_ = 'DT'
token.head.text = 'Moon'
token.dep_ = 'det'

TOKEN: Moon
=====
token.tag_ = 'NNP'
token.head.text = 'is'
token.dep_ = 'nsubj'

TOKEN: is
=====
token.tag_ = 'VBZ'
token.head.text = 'is'
token.dep_ = 'ROOT'

TOKEN: Earth
=====
token.tag_ = 'NNP'
token.head.text = 'satellite'
token.dep_ = 'poss'

TOKEN: 's
=====
token.tag_ = 'POS'
token.head.text = 'Earth'
token.dep_ = 'case'

TOKEN: only
=====
token.tag_ = 'JJ'
token.head.text = 'satellite'
token.dep_ = 'amod'

TOKEN: natural
=====
token.tag_ = 'JJ'
token.head.text = 'satellite'
token.dep_ = 'amod'

TOKEN: satellite
=====
token.tag_ = 'NN'
token.head.text = 'is'
token.dep_ = 'attr'

TOKEN: .
=====
token.tag_ = '.'
token.head.text = 'is'
token.dep_ = 'punct'

TOKEN: It
=====
token.tag_ = 'PRP'
token.head.text = 'orbits'
token.dep_ = 'nsubj'

TOKEN: orbits
=====
token.tag_ = 'VBZ'
token.head.text = 'orbits'
token.dep_ = 'ROOT'

TOKEN: at
=====
token.tag_ = 'IN'
token.head.text = 'orbits'
token.dep_ = 'prep'

TOKEN: an
=====
token.tag_ = 'DT'
token.head.text = 'distance'
token.dep_ = 'det'

TOKEN: average
=====
token.tag_ = 'JJ'
token.head.text = 'distance'
token.dep_ = 'amod'

TOKEN: distance
=====
token.tag_ = 'NN'
token.head.text = 'at'
token.dep_ = 'pobj'

TOKEN: of
=====
token.tag_ = 'IN'
token.head.text = 'distance'
token.dep_ = 'prep'

TOKEN: 384400
=====
token.tag_ = 'CD'
token.head.text = 'km'
token.dep_ = 'nummod'

TOKEN: km
=====
token.tag_ = 'NN'
token.head.text = 'of'
token.dep_ = 'pobj'

TOKEN: ,
=====
token.tag_ = ','
token.head.text = 'distance'
token.dep_ = 'punct'

TOKEN: about
=====
token.tag_ = 'RB'
token.head.text = '30'
token.dep_ = 'advmod'

TOKEN: 30
=====
token.tag_ = 'CD'
token.head.text = 'diameter'
token.dep_ = 'nummod'

TOKEN: times
=====
token.tag_ = 'NNS'
token.head.text = '30'
token.dep_ = 'quantmod'

TOKEN: Earth
=====
token.tag_ = 'NNP'
token.head.text = 'diameter'
token.dep_ = 'poss'

TOKEN: 's
=====
token.tag_ = 'POS'
token.head.text = 'Earth'
token.dep_ = 'case'

TOKEN: diameter
=====
token.tag_ = 'NN'
token.head.text = 'distance'
token.dep_ = 'appos'

TOKEN: .
=====
token.tag_ = '.'
token.head.text = 'orbits'
token.dep_ = 'punct'
'''