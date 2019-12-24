# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:02:36 2019

@author: Hans
"""

%reset -f
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#%% Read in the data, find out the header and the dimensions.
# In total there are 6335 news articles which are either fake or real that we can use to train upon

df=pd.read_csv('D:\\Git_Projects\\Fake News\\news.csv')
#Get shape and head
print(df.shape)
df.head()

#%% Extraxt the lables

label = df.label
label.head()

text = df.text
text.head()

#%% for this problem we will be using the sklearn package for 

#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(text, label, test_size=0.2)

""" for text analysis its worthwile to remove stopwords. These stopwords do not contribute to the news
 being fake or true. We can do this using the TfidfVectoriser which analysis the Term frequency and 
Inverse Document Frequency """

#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.8)
#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

#%% 
""" We implement the online passive agressive algortihm. When the label is correct, W is not changed. 
When it is wrong, W is changed aggrissively such that the label is forced to be correct. However in the minimal way.

"""

pac=PassiveAggressiveClassifier(C=0.01,max_iter = 500)
pac.fit(tfidf_train,y_train)
# Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

# CHeck false positives and negatives
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

# Set the y=0 labels to -1

