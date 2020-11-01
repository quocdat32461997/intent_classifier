"""
inference.py - module for intent-classification inference
"""

# import dependencies
import os
from sklearn.feature_extraction.text import TfidfVectorizer

class IntentClassifier:
	"""
	IntentClassifier - class for intent-classification
	"""

	def __init__(self, model, vocabs, intents):
		"""
		Constructor for IntentClassifier class
		Support hosting SVM model
		Inputs:
			- model : str or sklearn model
				Path to movel weights
			- vocabs : str or dict
				Path to dict or dictionary of keys (terms) and values (indices to mapping indices)
			- intents : str or list of intents
				Path to intent_list of list of intents
		"""

		# load model
		if isinstance(model, str):
			self.model = pickle.load(open(model, 'rb'))
		else:
			self.modle = model

		# load vocabs
		if isinstance(vocabs, str):
			self.vocabs = {}
			with open(vocabs) as file:
				words = file.read().split('\n')

				for idx in range(len(words)):
					self.vocabs[idx] = words[idx]

		else:
			self.vocabs = vocabs
					
		# load intents
		if isinstance(intents, str):
			with open(intents) as file:
				self.intents = file.read().split('\n')
		else:
			self.intents = intents
			

		# initlaize text-processing vectorizer
		_ = self._initialize_text_processor(vocabulary = vocabs)

	def _initialize_text_processer(self, vocabulary):
		"""
		_initialize_text_processor - function to initlaize text-processing pipeline by vectorizer
		Inputs:
			- vocabulary : list of str
				List of vocabs
		"""

		self.vectorizer = TfidfVectorizer(vocabulary = vocabulary)

	def process_text(self, text):
		"""
		Function to process text: lowercase, remove punctuations and stopwords
		"""

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
		output = self.intents[output]

		return output
		
