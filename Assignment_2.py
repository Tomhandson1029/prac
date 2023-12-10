'''
Assignment No.02
Name: Chinmay Bitne
Roll No: 14
Batch: B1
Title: "Assignment to implement Bag of Words and TFIDF using Gensim library."
'''

from gensim.utils import simple_preprocess
from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count
from gensim import corpora, models
import numpy as np

text = open('black_hole.txt', encoding ='utf-8')
 
tokens =[]
for line in text.read().split('.'):
  tokens.append(simple_preprocess(line, deacc = True))

g_dict = corpora.Dictionary(tokens)

print("The dictionary has: " +str(len(g_dict)) + " tokens\n")
print(g_dict.token2id)

g_bow =[g_dict.doc2bow(token, allow_update = True) for token in tokens]
print("Bag of Words : ", g_bow)

print("----------------------------------------------------------------------------")
text1 = ["The food is excellent but the service can be better", 
        "The food is always delicious and loved the service",
        "The food was mediocre and the service was terrible"]

g_dict = corpora.Dictionary([simple_preprocess(line) for line in text1])
g_bow = [g_dict.doc2bow(simple_preprocess(line)) for line in text1]

print("Dictionary : ")
for item in g_bow:
    print([[g_dict[id], freq] for id, freq in item])

g_tfidf = models.TfidfModel(g_bow, smartirs='ntc')

print("TF-IDF Vector:")
for item in g_tfidf[g_bow]:
    print([[g_dict[id], np.around(freq, decimals=2)] for id, freq in item])
print("----------------------------------------------------------------------------")
data = [
    "this is a sentence",
    "another example sentence",
    "word embeddings are useful",
    "word2vec is a popular model",
]

tokenized_data = [sentence.split() for sentence in data]

w2v_model = Word2Vec(tokenized_data, min_count=0, workers=cpu_count())

similar_words = w2v_model.wv.most_similar('word')

for word, score in similar_words:
    print(f"{word}: {score}")


'''
Output: 
//BOW
The dictionary has: 90 tokens

{'black': 0, 'electromagnetic': 1, 'energy': 2, 'enough': 3, 'escape': 4, 'gravity': 5, 'has': 6, 'hole': 7, 'including': 8, 'is': 9, 'it': 10, 'light': 11, 'nothing': 12, 'of': 13, 'or': 14, 'other': 15, 'region': 16, 'so': 17, 'spacetime': 18, 'strong': 19, 'that': 20, 'to': 21, 'waves': 22, 'where': 23, 'can': 24, 'compact': 25, 'deform': 26, 'form': 27, 'general': 28, 'mass': 29, 'predicts': 30, 'relativity': 31, 'sufficiently': 32, 'the': 33, 'theory': 34, 'boundary': 35, 'called': 36, 'event': 37, 'horizon': 38, 'no': 39, 'according': 40, 'although': 41, 'an': 42, 'and': 43, 'circumstances': 44, 'crossing': 45, 'detectable': 46, 'effect': 47, 'fate': 48, 'features': 49, 'great': 50, 'locally': 51, 'object': 52, 'on': 53, 'acts': 54, 'as': 55, 'body': 56, 'ideal': 57, 'in': 58, 'like': 59, 'many': 60, 'reflects': 61, 'ways': 62, 'curved': 63, 'emit': 64, 'field': 65, 'hawking': 66, 'horizons': 67, 'inversely': 68, 'its': 69, 'moreover': 70, 'proportional': 71, 'quantum': 72, 'radiation': 73, 'same': 74, 'spectrum': 75, 'temperature': 76, 'with': 77, 'billionths': 78, 'directly': 79, 'essentially': 80, 'for': 81, 'holes': 82, 'impossible': 83, 'kelvin': 84, 'making': 85, 'observe': 86, 'order': 87, 'stellar': 88, 'this': 89}
Bag of Words :  [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 2), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1), (21, 1), (22, 1), (23, 1)], [(0, 1), (7, 1), (13, 1), (18, 1), (20, 1), (21, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1), (29, 1), (30, 1), (31, 1), (32, 1), (33, 1), (34, 1)], [(4, 1), (9, 1), (13, 1), (33, 2), (35, 1), (36, 1), (37, 1), (38, 1), (39, 1)], [(6, 2), (10, 3), (13, 1), (21, 1), (28, 1), (31, 1), (33, 1), (39, 1), (40, 1), (41, 1), (42, 1), (43, 1), (44, 1), (45, 1), (46, 1), (47, 1), (48, 1), (49, 1), (50, 1), (51, 1), (52, 1), (53, 1)], [(0, 2), (7, 1), (10, 1), (11, 1), (39, 1), (42, 1), (54, 1), (55, 1), (56, 1), (57, 1), (58, 1), (59, 1), (60, 1), (61, 1), (62, 1)], [(0, 1), (13, 1), (18, 1), (20, 1), (21, 1), (29, 1), (30, 1), (33, 1), (34, 1), (37, 1), (55, 1), (56, 1), (58, 1), (63, 1), (64, 1), (65, 1), (66, 1), (67, 1), (68, 1), (69, 1), (70, 1), (71, 1), (72, 1), (73, 1), (74, 1), (75, 1), (76, 1), (77, 1)], [(0, 1), (9, 1), (10, 1), (13, 3), (21, 1), (33, 1), (76, 1), (78, 1), (79, 1), (80, 1), (81, 1), (82, 1), (83, 1), (84, 1), (85, 1), (86, 1), (87, 1), (88, 1), (89, 1)], []]

//TF-IDF
Dictionary : 
[['be', 1], ['better', 1], ['but', 1], ['can', 1], ['excellent', 1], ['food', 1], ['is', 1], ['service', 1], ['the', 2]]
[['food', 1], ['is', 1], ['service', 1], ['the', 2], ['always', 1], ['and', 1], ['delicious', 1], ['loved', 1]]
[['food', 1], ['service', 1], ['the', 2], ['and', 1], ['mediocre', 1], ['terrible', 1], ['was', 2]]
TF-IDF Vector:
[['be', 0.43], ['better', 0.43], ['but', 0.43], ['can', 0.43], ['excellent', 0.43], ['food', 0.09], ['is', 0.21], ['service', 0.09], ['the', 0.18]]
[['food', 0.11], ['is', 0.26], ['service', 0.11], ['the', 0.21], ['always', 0.52], ['and', 0.26], ['delicious', 0.52], ['loved', 0.52]]
[['food', 0.08], ['service', 0.08], ['the', 0.16], ['and', 0.2], ['mediocre', 0.39], ['terrible', 0.39], ['was', 0.78]]

//Word2Vec
----------------------------------------------------------------------------
sentence: 0.21617141366004944
embeddings: 0.044689226895570755
example: 0.015025208704173565
useful: 0.0019510718993842602
is: -0.03284316137433052
this: -0.04568909481167793
another: -0.0742427185177803
are: -0.09326908737421036
a: -0.09575342386960983
word2vec: -0.10513807833194733
'''
