import bs4 as bs
import urllib.request
import re
import nltk
import sys
from gensim.models import Word2Vec
import pandas as pd


def read_dataset(file,sheet,column1):
	df = pd.read_excel(str(file), sheet_name=str(sheet))
	URL = df['Url']
	print(URL)
	return URL


def read_article(URL):
	scrapped_data = urllib.request.urlopen(URL)
	article = scrapped_data .read()

	parsed_article = bs.BeautifulSoup(article,'lxml')

	paragraphs = parsed_article.find_all('p')

	article_text = ""

	for p in paragraphs:
	    article_text += p.text



	# Cleaing the text
	processed_article = article_text.lower()
	processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )
	processed_article = re.sub(r'\s+', ' ', processed_article)

	# Preparing the dataset
	all_sentences = nltk.sent_tokenize(processed_article)

	all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

	# Removing Stop Words
	from nltk.corpus import stopwords
	for i in range(len(all_words)):
	    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]

	return all_words


file = sys.argv[1]
sheet = sys.argv[2]
column = sys.argv[3]
print(file,sheet,column)
URL = read_dataset(file,sheet,column)
for i in range(5):
	print("\n\n"+str(i)+":      ")
	all_words = read_article(URL[i])

	word2vec = Word2Vec(all_words, min_count=10)
	vocabulary = word2vec.wv.vocab
	print(vocabulary)
