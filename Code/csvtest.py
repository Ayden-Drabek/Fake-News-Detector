import bs4 as bs
import urllib.request
import re
import nltk
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from adjustText import adjust_text
from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import PassiveAggressiveClassifier

import time

start_time = time.time()



def read_dataset():
	#df = pd.read_excel(str(file), sheet_name=str(sheet))
	#URL = df[column1]
	df = pd.read_csv("data.csv", error_bad_lines=False)
	Body = df['Body']
	label = df['Label']
	return Body, label


def read_article(URL):
	scrapped_data = urllib.request.urlopen(URL)
	article = scrapped_data .read()

	all_words = clean_article(article)
	return all_words


def clean_article(article):
	#parsed_article = bs.BeautifulSoup(article,'lxml')

	#paragraphs = parsed_article.find_all('p')
	'''article_text = ""

	for p in paragraphs:
	    article_text += p.text'''

	# Cleaing the text
	processed_article = str(article).lower()
	processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )
	processed_article = re.sub(r'\s+', ' ', processed_article)

	# Preparing the dataset
	all_sentences = nltk.sent_tokenize(processed_article)

	all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

	# Removing Stop Words
	from nltk.corpus import stopwords
	all_words = [w for w in all_words if w not in stopwords.words('english')]

	return all_words[0]

def word_to_vec(all_words):
	# Filter the list of vectors to include only those that Word2Vec has a vector for
	vector_list = [model[word] for word in all_words if word in model.vocab]
	avg = np.mean([model[word] for word in all_words if word in model.vocab])
	# Create a list of the words corresponding to these vectors
	words_filtered = [word for word in all_words if word in model.vocab]

	# Zip the words together with their vector representations
	word_vec_zip = zip(words_filtered, vector_list)

	# Cast to a dict so we can turn it into a DataFrame
	word_vec_dict = dict(word_vec_zip)

	df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
	return df


model = KeyedVectors.load_word2vec_format('/home/malu/GoogleNews-vectors-negative300.bin', binary=True)  
vocab = model.vocab.keys()
#file = sys.argv[1]


body, label = read_dataset()

avg = np.arange(300, dtype=np.float16)
data = []
newlabel = []

for i in range(3990):
	newlabel.append(label[i])
	#data = []
	#print(str(i))
	#print(body[i])
	all_words = clean_article(body[i])
	df = word_to_vec(all_words)
	for k in range(300):
		avg[k] = np.mean(df[k])

	#the average vector for each article 
	data.append(avg)
	#tsne(df)

#print(data)
print("vector label lenght  " + str(len(newlabel)))

print("data lenght  " + str(len(data)))
print("data column lenght  " + str(len(data[0])))

X_train, X_test, y_train, y_test = train_test_split(data, newlabel, test_size = 0.20)
#print(str(y_train) +"\n\n" + str(y_test) + "\n\n") 

svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print("------- SVM -------")
print(y_pred)


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(X_train,y_train)
#DataFlair - Predict on the test set and calculate accuracy
y_predPA=pac.predict(X_test)

score=accuracy_score(y_test,y_predPA)

print("------- PASSIVE AGRESSIVE CLASSIFIER -------")
print(y_predPA)

print(confusion_matrix(y_test, y_predPA))
print(classification_report(y_test, y_predPA))
print(f'Accuracy: {round(score*100,2)}%')


print("--- %s seconds ---" % (time.time() - start_time))
#plt.show()
