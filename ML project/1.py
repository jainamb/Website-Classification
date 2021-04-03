import os






import joblib
loaded_model = joblib.load('donebyme.pkl')
print(".....")


from googletrans import Translator

import io
import urllib.request as urllib2

import requests

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

import re


from sklearn.svm import LinearSVC
import io
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import string
import random
#ignore
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()





translator = Translator()


#this bag of wors will have all the text present in all urls(i.e total text)
def listToString(s):  
    # initialize an empty string
    str1 = ""  
   
    # traverse in the string  
    for ele in s:  
        str1 = str1 + " " + ele  
   
    # return string  
    return str1  
       
def Doall(string_url):

	soup = BeautifulSoup(string_url, features='html.parser')
	for script in soup(["script", "style"]):
		script.extract()    # rip it out
	text = soup.get_text()

	# break into lines and remove leading and trailing space on each
	lines = (line.strip() for line in text.splitlines())
	# break multi-headlines into a line each
	chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
	  # drop blank lines
	text = ', '.join(chunk for chunk in chunks if chunk)
	text = text.replace(',', ' ')
	text = text.replace(',', '')
	punctuations = '''!()-[]{};:'"\,|<>./?@#$%^&*_~'''

	# traverse the given string and if any punctuation
	# marks occur replace it with null
	for x in text.lower():
		if x in punctuations:
		  text = text.replace(x, "")

	l = text.split(" ")

	#removing ' ' elements
	done = [i for i in l if i != '']
	#convert to string
	done = [str (item) for item in done]

	# duplicates removed
	#stemming start
	ans = [ps.stem(w) for w in done]
	retrntext = listToString(ans)

	#stemming end
	#filling the bag of words

	return retrntext



def dataframpredict(f):

	temp = [[]]
	l = []
	j = 0
	print(j)
	j+=1
	tmn = Doall(f)
	if len(tmn) < 10:
		print(tmn)
		print("##############################")
		print('there was an issue...')
		print("##############################")
		print(tmn)
		return ["Text Not Available"]
	l.append(tmn)
	return l
print("...")


import codecs
import os


l = os.listdir("crawled/")
m = [n for n in l]
for i in range(0, len(m)):
	m[i] = "crawled/"+ m[i]+"/"


with open('55.txt', 'w', encoding='cp437', errors="ignore") as kk:

	for i in m:
		l = os.listdir(i)
		z = [i+t for t in l] 
		for j in z:
			print(j)
		
			f=codecs.open(j, 'r')
			try:

				xyz = dataframpredict(f)

				result = loaded_model.predict(xyz)
				result_confidence = loaded_model.predict_proba(xyz)
				print("..")
				print(result)
				again = str(j) + " , " + str(result[0])
				print(again)
				kk.write(again)
				kk.write("/n")


				print(result_confidence)

				Confidence_array = ["Arts & Entertainment", "Autos & Vehicles", "Beauty & Fitness", "Books & Literature",
				"Business & Industry", "Career and Education", "Comps & Electronics", "Finance", "Food & Drink", "Games", "Health", "Law & Government", "News & Media", "Pets & Aimals",
				"Recreation & Hobbies", "Reference", "Science", "Shopping", "Sports", "travel"]


				for i in range(0, len(result_confidence[0])):
					print(result_confidence[0][i], "-->", Confidence_array[i])

				dd = []
				for i in range(0, len(result_confidence[0])):
					dd.append([((result_confidence[0][i])/(sum(result_confidence[0])))*100, Confidence_array[i]])
				dd.sort()
				for i in range(len(result_confidence[0])-1, -1, -1):
					print(dd[i][1], " With a confidence level of -> ", dd[i][0])
			except:
			
				pass
	kk.close()