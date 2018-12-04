# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from unicodedata import normalize
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import os
import io


def pre_process(text):
    
    #Tranforma para minúsculo
    text=text.lower()
    
    #Remove dígitos e caracteres especiais
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text

def get_stop_words(stopwords):
    #Pega a lista de stopwords
    with open(stopwords, 'r') as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

def plot_classification_report(y_tru, y_prd, figsize=(10, 10), ax=None):
    #Função para plotar o classification report(https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report)
    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = list(np.unique(y_tru))
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep,
                annot=True, 
                cbar=False, 
                xticklabels=xticks, 
                yticklabels=yticks,
                ax=ax)

def data_to_dataframe():
	#Cria o dataframe dos dados, já pré-processados e adiciona a coluna label
	dir1 = 'Fake.br-Corpus/size_normalized_texts/true'
	dir2 = 'Fake.br-Corpus/size_normalized_texts/fake'
	filelist1 = os.listdir(dir1) 
	filelist2 = os.listdir(dir2)

	tf_idf1=pd.DataFrame(columns=['Texto'])
	for item in filelist1:
		with io.open(dir1+'/'+item, 'r', encoding='utf-8') as myfile:
    			data=myfile.read()
			data = normalize('NFKD', data).encode('ASCII', 'ignore').decode('ASCII')
			data=pre_process(data)
			tf_idf1=tf_idf1.append({'Texto' : data},ignore_index=True)

	
	tf_idf2=pd.DataFrame(columns=['Texto'])
	for item in filelist2:
		with io.open(dir2+'/'+item, 'r', encoding='utf-8') as myfile:
    			data=myfile.read()
			data = normalize('NFKD', data).encode('ASCII', 'ignore').decode('ASCII')
			data=pre_process(data)
			tf_idf2=tf_idf2.append({'Texto' : data},ignore_index=True)
	
	
	tf_idf1['Label'] = '1'
	tf_idf2['Label'] = '0'
	

	return (tf_idf1.append(tf_idf2))

def bayes(X_train,y_train,X_test,y_test,stopwords):
	#função para o classificador Multinomial Naive Bayes, utilizando pipeline e gridsearchcv
	text_clf = Pipeline([('vect', CountVectorizer(stop_words=stopwords,max_df=0.85,max_features=10000)), ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)), ('clf', MultinomialNB())])
	text_clf = text_clf.fit(X_train, y_train)
	parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
	gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1,cv=3)
	gs_clf = gs_clf.fit(X_train, y_train)
	print("Melhores parâmetros utilizando Multinomial Naive Bayes:""\n")
	print(gs_clf.best_params_)
	ypred = (gs_clf.best_estimator_).predict(X_test)
	print("\n""Acurácia de classificação média:")
	print(accuracy_score(y_test, ypred))
	print(classification_report(y_test, ypred))
	plot_classification_report(y_test, ypred)
	#Plota a matriz de confusão
	cm = confusion_matrix(y_test, ypred)
	plt.matshow(cm)
	plt.ylabel('Predict')
	plt.xlabel('True')
	plt.title('MATRIZ DE CONFUSAO Bayes')
	plt.colorbar()
	plt.show()
	


def SVM(X_train,y_train,X_test,y_test,stopwords):
	#função para o classificador SVM com treinamento usando SGD, utilizando pipeline e gridsearchcv
	text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words=stopwords,max_df=0.85,max_features=10000)), ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=500,tol=1e-3))])

	text_clf_svm = text_clf_svm.fit(X_train, y_train)
	parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}
	gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1,cv=3)
	gs_clf_svm = gs_clf_svm.fit(X_train, y_train)
	print("Melhores parâmetros utilizando SVM com SGD:")
	print(gs_clf_svm.best_params_)
	ypred2 = (gs_clf_svm.best_estimator_).predict(X_test)
	print("\n""Acurácia de classificação média:")
	print(accuracy_score(y_test, ypred2))
	print(classification_report(y_test, ypred2))
	plot_classification_report(y_test, ypred2)
	#Plota a matriz de confusão
	cm = confusion_matrix(y_test, ypred2)
	plt.matshow(cm)
	plt.ylabel('Predict')
	plt.xlabel('True')
	plt.title('MATRIZ DE CONFUSAO SVM')
	plt.colorbar()
	plt.show()
	

def main():
	tf_idf=data_to_dataframe()
	X=tf_idf['Texto']
	y=tf_idf['Label']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

	stopwords=get_stop_words("stopwords.txt")
	bayes(X_train,y_train,X_test,y_test,stopwords)
	SVM(X_train,y_train,X_test,y_test,stopwords)

if __name__ == "__main__":
    main()

