import bs4 as bs
import urllib.request
import re
import nltk
import sys
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

bankdata = pd.read_csv("data.csv")
body = bankdata['Body']
label = bankdata['Label']
dictlist = []
newlabel = []

for i in range(4):
	# Cleaing the text
	newlabel.append(label[i])
	processed_article = body[i].lower()
	processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )
	processed_article = re.sub(r'\s+', ' ', processed_article)

	# Preparing the dataset
	all_sentences = nltk.sent_tokenize(processed_article)

	all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

	# Removing Stop Words
	from nltk.corpus import stopwords
	for i in range(len(all_words)):
	    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]

	word2vec = Word2Vec(all_words, min_count=3)
	vocabulary = list(word2vec.wv.vocab)
	dictlist.append(vocabulary)
	

for x in range(len(dictlist)):
	print(str(dictlist[x]) +"\n" + str(newlabel[x]) + "\n\n") 


X_train, X_test, y_train, y_test = train_test_split(dictlist, newlabel, test_size = 0.20)
print(str(X_train) +"\n\n" + str(X_test) + "\n\n") 

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

'''y_pred = svclassifier.predict(X_test)


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))'''