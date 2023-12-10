'''
Assignment No.01
Name: Chinmay Bitne
Roll No: 14
Batch: B1
Title: "Text Pre-Processing using NLP operations: Perform Tokenization, Stop word removal, Lemmatization ,Part-of-Speech Tagging use any sample text"
'''

#import libraries
import spacy

# Load the language model
nlp = spacy.load("en_core_web_sm")

# Define the input text with spaces between sentences
about_text = (
   "I am interested in learning"
   " Natural Language Processing"
)

# 1. Tokenization:
about_doc = nlp(about_text)
print("1. Tokenization:")
for token in about_doc:
    print(token, token.idx)

# 2. Stop Words Removal:
about_doc = nlp(about_text)
print("\n2. Stop Words Removal:")
print([token for token in about_doc if not token.is_stop])

# 3. Lemmatization:
about_doc = nlp(about_text)
print("\n3. Lemmatization:")
for token in about_doc:
    if str(token) != str(token.lemma_):
        print(f"{str(token):>20} : {str(token.lemma_)}")

# 4. Part of Speech Tagging:
about_doc = nlp(about_text)
print("\n4. Part of Speech Tagging:")
for token in about_doc:
    print(
        f"""
TOKEN: {str(token)}
=====
TAG: {str(token.tag_):10} POS: {token.pos_}
EXPLANATION: {spacy.explain(token.tag_)}"""
    )

'''
===================Output===================
1. Tokenization:
I 0
am 2
interested 5
in 16
learning 19
Natural 28
Language 36
Processing 45

2. Stop Words Removal:
[interested, learning, Natural, Language, Processing]

3. Lemmatization:
                  am : be
            learning : learn
          Processing : processing

4. Part of Speech Tagging:

TOKEN: I
=====
TAG: PRP        POS: PRON
EXPLANATION: pronoun, personal

TOKEN: am
=====
TAG: VBP        POS: AUX
EXPLANATION: verb, non-3rd person singular present

TOKEN: interested
=====
TAG: JJ         POS: ADJ
EXPLANATION: adjective (English), other noun-modifier (Chinese)

TOKEN: in
=====
TAG: IN         POS: ADP
EXPLANATION: conjunction, subordinating or preposition

TOKEN: learning
=====
TAG: VBG        POS: VERB
EXPLANATION: verb, gerund or present participle

TOKEN: Natural
=====
TAG: NNP        POS: PROPN
EXPLANATION: noun, proper singular

TOKEN: Language
=====
TAG: NNP        POS: PROPN
EXPLANATION: noun, proper singular

TOKEN: Processing
=====
TAG: NN         POS: NOUN
EXPLANATION: noun, singular or mass
'''