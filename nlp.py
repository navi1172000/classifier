# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:29:14 2020

@author: naveen
"""

import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')

reviews=load_files('txt_sentoken/')
X,y=reviews.data,reviews.target

with open('X.pickle','wb') as f:
    pickle.dump(X,f)
    
with open('y.pickle','wb') as f:
    pickle.dump(y,f)


# Unpickling dataset
X_in = open('X.pickle','rb')
y_in = open('y.pickle','rb')
X = pickle.load(X_in)
y = pickle.load(y_in)


corpus=[]
for i in range(0,len(X)):
    reviews=re.sub(r'\W',' ',str(X[i]))
    reviews=reviews.lower()
    reviews=re.sub(r'\s+[a-z]\s+',' ',reviews)
    reviews=re.sub(r'\s+',' ',reviews)
    corpus.append(reviews)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()


from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(text_train,sent_train)


pred=classifier.predict(text_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(sent_test,pred)

from sklearn.metrics  import accuracy_score
acc=accuracy_score(y_true=sent_test,y_pred=pred)


with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    

with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)
    
    
with open('classifier.pickle','rb') as f:
    clf=pickle.load(f)
    

with open('tfidfmodel.pickle','rb') as f:
    tfidf=pickle.load(f)
    

sample=["You are a nice person man, have a good life"]
sample=tfidf.transform(sample).toarray()
print(clf.predict(sample))
