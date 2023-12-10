'''
Assignment No.03
Name: Chinmay Bitne
Roll No: 14
Batch: B1
Title: "Name Entity Recognition in python with spacy"
'''

import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def perform_ner(text):
    # Process the text using SpaCy
    doc = nlp(text)
    
    # Extract named entities and their labels
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return entities

if __name__ == "__main__":
    # Example text
    text = "The Moon is Earth's only natural satellite. It orbits at an average distance of 384400 km, about 30 times Earth's diameter."

    # Perform Named Entity Recognition
    named_entities = perform_ner(text)

    # Print the results
    print("Named Entities:")
    for entity, label in named_entities:
        print(f"{entity} - {label}")

'''
Output:

Named Entities:
Moon - PERSON
Earth - LOC
384400 km - QUANTITY
about 30 - CARDINAL
Earth - LOC
'''