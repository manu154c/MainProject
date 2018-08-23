"""
@Auther : MANU C
Created : 15-03-18
Last Modified : 08-08-18

Title : Skip-Gram Word2Vec Implementation

Status : Bigram Model 

Next Work : Convert to skip-gram

Libraries Used : NLTK, NUMPY

Python 3.6

"""

from django.shortcuts import render


import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
import sklearn.datasets as skds
from pathlib import Path

# Debugging
import pdb

# Create your views here.

def index(request):
	data = pre_processing()
	dict_train_and_test = split_data_for_training_and_testing(data)

	if request.method == "POST":
		output = ""
		return render(request, 'the_feature_learning_network/index.html', {'output' : output})
	else:
		return render(request, 'the_feature_learning_network/index.html')


def pre_processing():

	#for reproducibility
	np.random.seed(1237)

	#source file directory
	path_train = "/home/manu/Downloads/20news-bydate/20news-bydate-train/"

	files_train = skds.load_files(path_train,load_content=False)

	label_index = files_train.target
	label_names = files_train.target_names
	labelled_files = files_train.filenames

	data_tags = ["filename","category","news"]
	data_list = []

	# Read and add data from file to a list
	i=0
	for f in labelled_files:
		ch = open(f, 'rb')
		data_list.append((f,label_names[label_index[i]], ch))
		ch.close()
		i += 1

	#pdb.set_trace()

	# We have training data available as dictionary filename, category, data
	data = pd.DataFrame.from_records(data_list, columns=data_tags)

	return data


def split_data_for_training_and_testing(data):

	# lets take 80% data as training and remaining 20% for test.
	train_size = int(len(data) * .8)

	output = {}

	output['train_posts'] = data['news'][:train_size]
	output['train_tags'] = data['category'][:train_size]
	output['train_files_names'] = data['filename'][:train_size]

	output['test_posts'] = data['news'][train_size:]
	output['test_tags'] = data['category'][train_size:]
	output['test_files_names'] = data['filename'][train_size:]

	return output

def tokenizing_and_vocabulary(train_posts, test_posts, train_tags, test_tags):
	# 20 news groups
	num_labels = 20
	vocab_size = 15000
	batch_size = 100

	#define Tokenizer with Vocab size
	tokenizer = Tokenizer(num_words=vocab_size)
	tokenizer.fit_on_texts(train_posts)

	x_train = tokenizer.text_to_matrix(train_posts, mode='tfidf')
	x_text = tokenizer.text_to_matrix(test_posts, mode='tfidf')

	encoder = LabelBinarizer()
	encoder.fit(train_tags)
	y_train = encoder.transform(train_tags)
	y_test = encoder.transform(test_tags)


