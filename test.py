"""
test.py - testing module
"""

# import dependencies

from inference import IntentClassifier

# define paths to model, vocabs, and intents
model = 'intent_classifier.sav'
intents = 'intent_list.txt'
vocabs = 'vocabs.pickle'
vectorizer = 'tfidf_vectorizer.pickle'

model = IntentClassifier(model, intents = intents, vocabs = vocabs, vectorizer = vectorizer)

model.predict("Who am I")
