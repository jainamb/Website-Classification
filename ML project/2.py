!pip install googletrans
!pip install requests
from googletrans import Translator

from google.colab import files

uploaded = files.upload()
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



stopwords = ['a', 'about', 'above', 'across', 'after',
 'afterwards', 'again', 'against', 'all', 'almost', 'alone',
'along', 'already', 'also', 'although', 'always', 'am',
'among', 'amongst', 'amoungst', 'amount',
'an', 'and', 'another', 'any', 'anyhow', 'anyone',
 'anything', 'anyway', 'anywhere', 'are', 'around',
'as', 'at', 'back', 'be', 'became', 'because', 'become',
 'becomes', 'becoming', 'been', 'before',
'beforehand', 'behind', 'being', 'below', 'beside',
'besides', 'between', 'beyond', 'bill', 'both',
'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant',
 'co', 'con', 'could', 'couldnt', 'cry', 'de',
'describe', 'detail', 'did', 'do', 'does', 'doing', 'don',
 'done', 'down', 'due', 'during', 'each', 'eg',
'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty',
 'enough', 'etc', 'even', 'ever', 'every', 'everyone',
'everything', 'everywhere', 'except', 'few', 'fifteen',
 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for',
'former', 'formerly', 'forty', 'found', 'four', 'from',
 'front', 'full', 'further', 'get', 'give', 'go', 'had',
'has', 'hasnt', 'have', 'having', 'he', 'hence', 'her',
 'here', 'hereafter', 'hereby', 'herein', 'hereupon',
'hers', 'herself', 'him', 'himself', 'his', 'how', 'however',
'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed',
'interest', 'into', 'is', 'it', 'its', 'itself', 'just',
 'keep', 'last', 'latter', 'latterly', 'least', 'less',
'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might',
 'mill', 'mine', 'more', 'moreover', 'most', 'mostly',
'move', 'much', 'must', 'my', 'myself', 'name', 'namely',
 'neither', 'never', 'nevertheless', 'next', 'nine',
'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing',
 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once',
'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise',
'our', 'ours', 'ourselves', 'out', 'over', 'own',
'part', 'per', 'perhaps', 'please', 'put', 'rather', 're',
 's', 'same', 'see', 'seem', 'seemed', 'seeming',
'seems', 'serious', 'several', 'she', 'should', 'show', 'side',
'since', 'sincere', 'six', 'sixty', 'so',
'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes',
 'somewhere', 'still', 'such', 'system',
't', 'take', 'ten', 'than', 'that', 'the', 'their', 'theirs',
 'them', 'themselves', 'then', 'thence', 'there',
'thereafter', 'thereby', 'therefore', 'therein', 'thereupon',
 'these', 'they', 'thickv', 'thin', 'third', 'this',
'those', 'though', 'three', 'through', 'throughout', 'thru',
'thus', 'to', 'together', 'too', 'top', 'toward',
'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until',
 'up', 'upon', 'us', 'very', 'via', 'was', 'we',
'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever',
 'where', 'whereafter', 'whereas', 'whereby',
'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
 'whither', 'who', 'whoever', 'whole', 'whom',
'whose', 'why', 'will', 'with', 'within', 'without', 'would',
'yet', 'you', 'your', 'yours', 'yourself',
'yourselves']

names=['URL','Category']


df=pd.read_csv(io.StringIO(uploaded['over.csv'].decode('utf-8')),names=names, na_filter=False)

df1 = df[250:300]
df2 = df[750:800]
df3 = df[1250:1300]
df4 = df[1750:1800]
df5 = df[2250:2300]
df6 = df[2750:2800]
df7 = df[3250:3300]
df8 = df[3750:3800]
df9 = df[4250:4300]
df10 = df[4750:4800]
df11 = df[5250:5300]
df12 = df[5750:5800]
df13 = df[6250:6300]
df14 = df[6700:6725]
df15 = df[7180:7220]
df16 = df[7500:7540]
df17 = df[7800:7850]
df18 = df[8300:8350]
df19 = df[8800:8850]
df20 = df[9300:9350]
dt=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20], axis=0)

df.drop(df.index[250:300],inplace= True)
df.drop(df.index[750:800],inplace= True)
df.drop(df.index[1250:1300],inplace= True)
df.drop(df.index[1750:1800],inplace= True)
df.drop(df.index[2250:2300],inplace= True)
df.drop(df.index[2750:2800],inplace= True)
df.drop(df.index[3250:3300],inplace= True)
df.drop(df.index[3750:3800],inplace= True)
df.drop(df.index[4250:4300],inplace= True)
df.drop(df.index[4750:4800],inplace= True)
df.drop(df.index[5250:5300],inplace= True)
df.drop(df.index[5750:5800],inplace= True)
df.drop(df.index[6250:6300],inplace= True)
df.drop(df.index[6700:6725],inplace= True)
df.drop(df.index[7180:7220],inplace= True)
df.drop(df.index[7500:7540],inplace= True)
df.drop(df.index[7800:7850],inplace= True)
df.drop(df.index[8300:8350],inplace= True)
df.drop(df.index[8800:8850],inplace= True)
df.drop(df.index[9300:9350],inplace= True)



df.tail()



###################################################################################

bag_of_words = []
def wait_for_internet_connection():
  while True:
    try:
      print("...")
      response = urllib2.urlopen('https://www.amazon.in/',timeout=3)
      return
    except urllib2.URLError:
      pass


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
  if string_url[0:3] == "htt":
    url_1 = string_url
    url_2 = string_url
  else:
    url_1 = "https://"+string_url
    url_2 = "http://"+string_url

  try:
    try:
      wait_for_internet_connection()
      print("opening this url")
      print(url_1)

      source = requests.get(url_1, timeout = 2).text
      soup = BeautifulSoup(source, features='lxml')
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
      mn = [translator.translate(k) for k in l]
      #removing ' ' elements
      done = [i for i in mn if i != '']
      #convert to string
      done = [str (item) for item in done]


      for i in range(0, len(done)):
        if done[i] not in no_duplicates:
          no_duplicates.append(done[i])
      # duplicates removed
      #stemming start
      ans = [ps.stem(w) for w in done]
      retrntext = listToString(ans)

      #stemming end
      #filling the bag of words

      return retrntext
    except:
      wait_for_internet_connection()
      print("opening this url")
      print(url_2)
      source = requests.get(url_2, timeout = 2).text
      soup = BeautifulSoup(source, features='lxml')
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
      #stemming end
      #stemming end
      #filling the bag of words
      retrntext = listToString(ans)
      return retrntext
  except:
    print("cannot open this url")
    print(string_url)
    #if url was not succesfully opened
    return " "
    pass






def datafram(datafr):
  length_of_data_fram = len(datafr)
  temp = [[]]
  l = []
  j = 0
  for i in datafr:
    print(j)
    j+=1
    l.append(Doall(i))

  return l

###############################################################################################

################################################################################################
def doalltest(string_url):


  if string_url[0:3] == "htt":
    url_1 = string_url
    url_2 = string_url
  else:
    url_1 = "https://"+string_url
    url_2 = "http://"+string_url

  try:
    try:
      wait_for_internet_connection()
      print("opening this url")
      print(url_1)

      source = requests.get(url_1, timeout = 2).text
      soup = BeautifulSoup(source, features='lxml')
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


      for i in range(0, len(done)):
        if done[i] not in no_duplicates:
          no_duplicates.append(done[i])
      # duplicates removed
      #stemming start
      ans = [ps.stem(w) for w in done]
      retrntext = listToString(ans)

      #stemming end
      #filling the bag of words

      return retrntext
    except:
      wait_for_internet_connection()
      print("opening this url")
      print(url_2)
      source = requests.get(url_2, timeout = 2).text
      soup = BeautifulSoup(source, features='lxml')
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
      #stemming end

      #stemming end
      #filling the bag of words
      retrntext = listToString(ans)
      return retrntext
  except:
    print("cannot open this url")
    print(string_url)
    #if url was not succesfully opened

    return " "
    pass


def dataframtest(datafr):
  length_of_data_fram = len(datafr)
  temp = [[]]
  l = []
  j = 0
  another = 0
  for i in datafr:
    print(j)
    j+=1
    some_other = Doall(i)
    if some_other == " ":
      another+=1
    l.append(some_other)
  return [l, another]


from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}

x_training = datafram(df['URL'])
y_training = df['Category']
from sklearn import svm
svc = svm.SVC()
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)



gs_clf = gs_clf.fit(x_training, y_training)

X_testing = dataframtest(dt['URL'])
calculatr = X_testing[1]
real = X_testing[0]
y_test=dt['Category']
print(len(y_test))

gs_clf.best_score_
gs_clf.best_params_
#training
predicted_svm = gs_clf.predict(real)
some = np.mean(predicted_svm == y_test)
print(some)
print((len(y_test.index)*some)/(len(y_test.index)-calculatr))






Confidence_array = ["Adult", "Arts & Entertainment", "Autos & Vehicles", "Beauty & Fitness", "Books & Literature",
"Business & Industry", "Career and Education", "Comps & Electronics", "Finance", "Food & Drink",
"Gambling", "Games", "Health", "Internet & Telecom", "Law & Government", "News & Media", "People & Society", "Pets & Aimals",
"Recreation & Hobbies", "Reference", "Science", "Shopping", "Sports", "travel"]

def dataframpredict(datafr):
  length_of_data_fram = len(datafr)
  temp = [[]]
  l = []
  j = 0
  print(j)
  j+=1
  l.append(Doall(datafr))
  return l

xyz = input()
xyz = dataframpredict(xyz)
print(gs_clf.predict(xyz))

from sklearn.externals import joblib
joblib.dump(gs_clf, 'donebyme.pkl')