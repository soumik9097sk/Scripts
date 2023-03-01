# TFIDF from scratch 
import pandas as pd
import numpy as np
import nltk

from nltk import wordtokenize

nltk.download('punkt')

# converting document to index dictionary
document = "list of string"
tokens = document.split() # or any tokenizer
count = 0
wordtoindex = {}
tokenized_docs = []
docs_as_int = =[]
for token in tokens:
	if token not in wordtoindex.keys():
		wordtoindex[token] = count
		count += 1

		# save for later 
		docs_as_int.append(wordtoindex[token]) # appending indexes
	tokenized_docs.append(docs_as_int) # appending 	


#mapping index to words
indextowords = {v,i for i,v in wordtoindex.items()}

N = len(df['text'])
V = len(wordtoindex)

tf = np.zeroes((N,V))

for i, docs_as_int in enumerate(tokenized_docs):
	for j in docs_as_int:
		tf[i,j] += j

document_freq = np.sum(tf>0, axis = 0) # document frequency
idf = np.log(N/document_freq)

# COMPUTING TFIDF VALUES
tf_idf = tf*idf # numpy broadcasting doing (N,V) , (V) multiplication



