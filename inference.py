"""
inference.py - module for intent-classification inference
"""

# import dependencies
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class IntentClassifier:
	"""
	IntentClassifier - class for intent-classification
	"""

	def __init__(self, model, vectorizer, intents):
		"""
		Constructor for IntentClassifier class
		Support hosting SVM model
		Inputs:
			- model : str or sklearn model
				Path to movel weights
			- vectorizer : str or Sklearn Vectorizer
				Path to vectorizer-file or Sklearn Vectorizer object
			- intents : str or list of intents
				Path to intent_list of list of intents
		"""

		# load model
		print("Loading IntentClassifier model")
		if isinstance(model, str):
			self.model = pickle.load(open(model, 'rb'))
		else:
			self.modle = model

		# load intents
		if isinstance(intents, str):
			with open(intents) as file:
				self.intents = file.read().split('\n')
		else:
			self.intents = intents
			

		# load vectorizer
		if isinstance(vectorizer, str):
			self.vectorizer = pickle.load(open(vectorizer, 'rb'))
		else:
			self.vectorizer = vectorizer

	def process_text(self, text):
		"""
		Function to process text: lowercase, remove punctuations and stopwords
		"""

		# convert to list
		if isinstance(text, str):
			text = [text]

		return self.vectorizer.transform(text)

	def predict(self, input):
		"""
		prediction - function to make predictions
		Inputs:
			- input : str
				Raw text
		Outputs: 
			- output : str
				Correct intent tag
		"""

		# extract TF-iDF features
		input = self.process_text(input)

		# make predictions
		output = self.model.predict(input)

		# parse raw-predictions to correct intent class
		output = self.intents[output[0]]

		return output
		
