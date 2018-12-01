# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from unicodedata import normalize
import re
import os
import io

dir1 = 'Fake.br-Corpus/size_normalized_texts/true'
dir2 = 'Fake.br-Corpus/size_normalized_texts/fake'
filelist1 = os.listdir(dir1) 
filelist2 = os.listdir(dir2)
tf_idf1=pd.DataFrame(columns=['Texto'])
tf_idf2=pd.DataFrame(columns=['Texto'])

def pre_process(text):
    
    #Tranforma para minúsculo
    text=text.lower()
    
    #Remove dígitos e caracteres especiais
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text

def get_stop_words(stopwords):

    with open(stopwords, 'r') as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

for item in filelist1:
	with io.open(dir1+'/'+item, 'r', encoding='utf-8') as myfile:
    		data=myfile.read()
		data = normalize('NFKD', data).encode('ASCII', 'ignore').decode('ASCII')
		data=pre_process(data)
		tf_idf1=tf_idf1.append({'Texto' : data},ignore_index=True)

for item in filelist2:
	with io.open(dir2+'/'+item, 'r', encoding='utf-8') as myfile:
    		data=myfile.read()
		data = normalize('NFKD', data).encode('ASCII', 'ignore').decode('ASCII')
		data=pre_process(data)
		tf_idf2=tf_idf2.append({'Texto' : data},ignore_index=True)
	
print(tf_idf1.shape)
print(tf_idf1.head(5))
print(tf_idf2.shape)
print(tf_idf2.head(5))
tf_idf1['Label'] = '1'
tf_idf2['Label'] = '0'
print(tf_idf1.shape)
print(tf_idf1.head(5))
print(tf_idf2.shape)
print(tf_idf2.head(5))

tf_idf=tf_idf1.append(tf_idf2)

X=tf_idf['Texto']
y=tf_idf['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print (X_test.shape, y_test.shape)

stopwords=get_stop_words("stopwords.txt")

text_clf = Pipeline([('vect', CountVectorizer(stop_words=stopwords,max_df=0.85,max_features=10000)), ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)), ('clf', MultinomialNB())])

text_clf = text_clf.fit(X_train, y_train)
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1,cv=3)
gs_clf = gs_clf.fit(X_train, y_train)
print(gs_clf.best_score_)
print(gs_clf.best_params_)

text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words=stopwords,max_df=0.85,max_features=10000)), ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42))])

text_clf_svm = text_clf_svm.fit(X_train, y_train)
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}
gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1,cv=3)
gs_clf_svm = gs_clf_svm.fit(X_train, y_train)


print(gs_clf_svm.best_score_)
print(gs_clf_svm.best_params_)

